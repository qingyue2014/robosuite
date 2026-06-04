"""
VLA model wrappers for PhysCogSafe L1 evaluation.

Supported:
  --model openvla    →  OpenVLA-7B  (openvla/openvla-7b on HuggingFace)
  --model pi0        →  π0          (physical_intelligence/pi0, requires separate install)
  --model random     →  random baseline (default, no GPU needed)

Usage:
    model = load_model("openvla", device="cuda")
    action = model.predict(image_rgb_np, instruction_str)
"""

import abc

import numpy as np


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class VLAModelBase(abc.ABC):
    @abc.abstractmethod
    def predict(self, image_rgb: np.ndarray, instruction: str) -> np.ndarray:
        """
        Args:
            image_rgb: uint8 RGB image, shape (H, W, 3)
            instruction: natural-language task string
        Returns:
            action: float32 array of shape (7,), values in [-1, 1]
                    [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """

    def reset(self):
        """Called at the start of each episode. Override if model is stateful."""
        pass


# ---------------------------------------------------------------------------
# Random baseline (no model, no GPU)
# ---------------------------------------------------------------------------

class RandomModel(VLAModelBase):
    def __init__(self, action_dim=7, seed=None):
        self._rng = np.random.default_rng(seed)

    def predict(self, image_rgb, instruction):
        return self._rng.uniform(-1, 1, size=7).astype(np.float32)


# ---------------------------------------------------------------------------
# OpenVLA
# ---------------------------------------------------------------------------

class OpenVLAModel(VLAModelBase):
    """
    OpenVLA-7B wrapper.

    Install:
        pip install transformers accelerate timm
        # Optional for faster inference:
        pip install flash-attn --no-build-isolation

    Model card: https://huggingface.co/openvla/openvla-7b

    Action format: 7-DoF delta end-effector + gripper, range [-1, 1].
    The unnorm_key selects which dataset's action statistics to use for
    de-normalisation. Use "bridge_orig" for table-top pick-and-place tasks
    closest to robosuite's setup. Override via --unnorm_key if needed.
    """

    def __init__(self, device="cuda", unnorm_key="bridge_orig", load_in_8bit=False):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        self.unnorm_key = unnorm_key

        print(f"[OpenVLA] Loading openvla/openvla-7b on {device} ...")
        dtype = torch.bfloat16

        load_kwargs = dict(
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["device_map"] = device

        self.model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b", **load_kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        )
        print("[OpenVLA] Model loaded.")

    def predict(self, image_rgb, instruction):
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(
            images=pil_image,
            text=instruction,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )
        # action is a numpy array of shape (7,)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# π0 (pi_zero) — Physical Intelligence
# ---------------------------------------------------------------------------

class Pi0Model(VLAModelBase):
    """
    π0 wrapper.

    Install:
        pip install lerobot
        # or follow https://github.com/huggingface/lerobot instructions

    π0 is available via the lerobot library as:
        lerobot/pi0  (on HuggingFace)

    Note: π0 operates in an action-chunking mode and is stateful across steps.
    Call reset() at episode start. The chunk_size controls how many steps ahead
    the model plans; default is 16.
    """

    def __init__(self, device="cuda", chunk_size=16):
        try:
            from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
            from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
            import torch
        except ImportError:
            raise ImportError(
                "π0 requires lerobot: pip install lerobot\n"
                "See https://github.com/huggingface/lerobot for setup."
            )

        self.device = device
        self.chunk_size = chunk_size
        self._action_queue = []

        print(f"[π0] Loading lerobot/pi0 on {device} ...")
        self.policy = PI0Policy.from_pretrained("lerobot/pi0").to(device)
        self.policy.eval()
        print("[π0] Model loaded.")

    def reset(self):
        self._action_queue = []
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, image_rgb, instruction):
        import torch
        from PIL import Image

        # If we have queued actions from a previous chunk, return next one
        if self._action_queue:
            return self._action_queue.pop(0)

        # Otherwise, run model to get next chunk of actions
        pil_image = Image.fromarray(image_rgb)
        # lerobot policy expects a specific observation dict format
        obs = {
            "observation.images.top": torch.from_numpy(
                np.array(pil_image).transpose(2, 0, 1)
            ).float().unsqueeze(0).to(self.device) / 255.0,
            "observation.state": torch.zeros(1, 7).to(self.device),
            "task": [instruction],
        }
        with torch.no_grad():
            actions = self.policy.select_action(obs)  # (chunk_size, 7)

        actions_np = actions.cpu().numpy()
        self._action_queue = [np.clip(a, -1.0, 1.0).astype(np.float32)
                               for a in actions_np[1:]]  # queue remaining
        return np.clip(actions_np[0], -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(model_name: str, **kwargs) -> VLAModelBase:
    """
    Load a VLA model by name.

    Args:
        model_name: "random" | "openvla" | "pi0"
        **kwargs: passed to the model constructor
                  e.g. device="cuda", unnorm_key="bridge_orig"
    """
    model_name = model_name.lower()
    if model_name == "random":
        return RandomModel(**kwargs)
    elif model_name == "openvla":
        return OpenVLAModel(**kwargs)
    elif model_name in ("pi0", "pi_zero"):
        return Pi0Model(**kwargs)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: random, openvla, pi0"
        )
