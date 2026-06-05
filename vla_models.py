"""
VLA model wrappers for PhysCogSafe L1 evaluation.

Supported:
  --model openvla    →  OpenVLA-7B  (openvla/openvla-7b on HuggingFace)
  --model openvla_oft → OpenVLA-OFT (requires moojink/openvla-oft on PYTHONPATH)
  --model pi0        →  π0          (physical_intelligence/pi0, requires separate install)
  --model random     →  random baseline (default, no GPU needed)

Usage:
    model = load_model("openvla", device="cuda")
    action = model.predict(image_rgb_np, instruction_str)
"""

import abc
import multiprocessing as mp
import traceback

import numpy as np

from robosuite.utils.transform_utils import quat2axisangle


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
            action: float32 array of shape (7,), values in the target
                    environment's action space.
        """

    def reset(self):
        """Called at the start of each episode. Override if model is stateful."""
        pass

    def close(self):
        """Called before process exit if the model owns external resources."""
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

    Action format: 7-DoF delta end-effector + gripper.
    The unnorm_key selects which dataset's action statistics to use for
    de-normalisation. Use "bridge_orig" for table-top pick-and-place tasks
    closest to robosuite's setup. Override via --unnorm_key if needed.
    """

    def __init__(
        self,
        device="cuda",
        unnorm_key="bridge_orig",
        load_in_8bit=False,
        controller_delta_scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        invert_gripper=False,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        self.unnorm_key = unnorm_key
        self.controller_delta_scale = np.array(controller_delta_scale, dtype=np.float32)
        self.invert_gripper = invert_gripper

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

    @staticmethod
    def _format_prompt(instruction):
        instruction = instruction.strip()
        return f"In: What action should the robot take to {instruction}?\nOut:"

    def _to_robosuite_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Expected OpenVLA action shape (7,), got {action.shape}")

        robosuite_action = np.empty(7, dtype=np.float32)
        robosuite_action[:6] = action[:6] / self.controller_delta_scale
        robosuite_action[6] = -action[6] if self.invert_gripper else action[6]
        return np.clip(robosuite_action, -1.0, 1.0).astype(np.float32)

    def predict(self, image_rgb, instruction):
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image_rgb)
        prompt = self._format_prompt(instruction)
        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )
        return self._to_robosuite_action(action)


# ---------------------------------------------------------------------------
# OpenVLA-OFT
# ---------------------------------------------------------------------------

class OpenVLAOFTModel(VLAModelBase):
    """
    OpenVLA-OFT wrapper.

    Install the official repository separately and expose it on PYTHONPATH:
        git clone https://github.com/moojink/openvla-oft.git /path/to/openvla-oft
        export PYTHONPATH=/path/to/openvla-oft:$PYTHONPATH

    Public OFT checkpoints are mainly LIBERO fine-tunes. This wrapper adapts
    robosuite observations to OFT's expected observation dict, but task success
    still depends on checkpoint / task / action-space compatibility.
    """

    def __init__(
        self,
        device="cuda",
        pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
        unnorm_key="libero_spatial_no_noops",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        load_in_8bit=False,
        load_in_4bit=False,
        controller_delta_scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        invert_gripper=False,
    ):
        try:
            import torch
            from experiments.robot.libero.run_libero_eval import GenerateConfig
            from experiments.robot.openvla_utils import (
                get_action_head,
                get_noisy_action_projector,
                get_processor,
                get_proprio_projector,
                get_vla,
                get_vla_action,
            )
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
        except ImportError as exc:
            raise ImportError(
                "OpenVLA-OFT requires the official repository on PYTHONPATH.\n"
                "Example:\n"
                "  git clone https://github.com/moojink/openvla-oft.git /path/to/openvla-oft\n"
                "  export PYTHONPATH=/path/to/openvla-oft:$PYTHONPATH\n"
                "Then install its requirements from SETUP.md."
            ) from exc

        # Official OFT utilities use their own module-level CUDA device. Keep
        # this argument for CLI consistency, but select GPUs with CUDA_VISIBLE_DEVICES.
        del device
        self.cfg = GenerateConfig(
            pretrained_checkpoint=pretrained_checkpoint,
            use_l1_regression=use_l1_regression,
            use_diffusion=use_diffusion,
            use_film=use_film,
            num_images_in_input=num_images_in_input,
            use_proprio=use_proprio,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            center_crop=center_crop,
            num_open_loop_steps=num_open_loop_steps or NUM_ACTIONS_CHUNK,
            unnorm_key=unnorm_key,
        )
        self.get_vla_action = get_vla_action
        self.controller_delta_scale = np.array(controller_delta_scale, dtype=np.float32)
        self.invert_gripper = invert_gripper
        self._action_queue = []

        print(f"[OpenVLA-OFT] Loading {pretrained_checkpoint} ...")
        self.vla = get_vla(self.cfg)
        self.processor = get_processor(self.cfg)
        self.action_head = None
        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        self.proprio_projector = None
        if self.cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                self.cfg,
                llm_dim=self.vla.llm_dim,
                proprio_dim=PROPRIO_DIM,
            )
        self.noisy_action_projector = None
        if self.cfg.use_diffusion:
            self.noisy_action_projector = get_noisy_action_projector(self.cfg, llm_dim=self.vla.llm_dim)
        torch.cuda.empty_cache()
        print("[OpenVLA-OFT] Model loaded.")

    def reset(self):
        self._action_queue = []

    @staticmethod
    def _obs_to_proprio(obs):
        pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
        quat = np.asarray(obs.get("robot0_eef_quat_site", obs.get("robot0_eef_quat", [0, 0, 0, 1])), dtype=np.float32)
        axis_angle = quat2axisangle(quat).astype(np.float32)
        gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", [0.0]), dtype=np.float32).reshape(-1)
        gripper = np.array([float(np.mean(gripper_qpos))], dtype=np.float32)
        return np.concatenate([pos, axis_angle, gripper, np.zeros(1, dtype=np.float32)]).astype(np.float32)

    def _to_robosuite_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 7:
            raise ValueError(f"Expected OpenVLA-OFT action with at least 7 dims, got {action.shape}")
        action = action[:7]
        robosuite_action = np.empty(7, dtype=np.float32)
        robosuite_action[:6] = action[:6] / self.controller_delta_scale
        robosuite_action[6] = -action[6] if self.invert_gripper else action[6]
        return np.clip(robosuite_action, -1.0, 1.0).astype(np.float32)

    def predict_from_obs(self, obs, instruction, image_key="agentview_image", wrist_image_key=None):
        if self._action_queue:
            return self._action_queue.pop(0)

        image = obs[image_key]
        observation = {
            "full_image": image,
            "task_description": instruction,
        }
        if self.cfg.num_images_in_input > 1:
            observation[wrist_image_key or "wrist_image"] = obs.get(wrist_image_key, image) if wrist_image_key else image
        if self.cfg.use_proprio:
            observation["state"] = self._obs_to_proprio(obs)

        actions = self.get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            observation,
            instruction,
            self.action_head,
            self.proprio_projector,
            self.noisy_action_projector,
            use_film=self.cfg.use_film,
        )
        converted = [self._to_robosuite_action(action) for action in actions]
        self._action_queue = converted[1 : self.cfg.num_open_loop_steps]
        return converted[0]

    def predict(self, image_rgb, instruction):
        return self.predict_from_obs({"agentview_image": image_rgb}, instruction)


def _model_worker(conn, model_cls, kwargs):
    try:
        model = model_cls(**kwargs)
        conn.send(("ready", None))
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "predict":
                _, image_rgb, instruction = msg
                action = model.predict(image_rgb, instruction)
                conn.send(("ok", action))
            elif cmd == "predict_from_obs":
                _, obs, instruction, image_key, wrist_image_key = msg
                if hasattr(model, "predict_from_obs"):
                    action = model.predict_from_obs(
                        obs,
                        instruction,
                        image_key=image_key,
                        wrist_image_key=wrist_image_key,
                    )
                else:
                    action = model.predict(obs[image_key], instruction)
                conn.send(("ok", action))
            elif cmd == "reset":
                model.reset()
                conn.send(("ok", None))
            elif cmd == "close":
                conn.send(("ok", None))
                break
            else:
                raise ValueError(f"Unknown model worker command: {cmd}")
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class SubprocessModel(VLAModelBase):
    """Run a CUDA VLA in a separate process to isolate Torch CUDA from MuJoCo EGL."""

    def __init__(self, model_cls, **kwargs):
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(target=_model_worker, args=(child_conn, model_cls, kwargs), daemon=True)
        self._proc.start()
        status, payload = self._parent_conn.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"Model subprocess failed to start:\n{payload}")

    def _request(self, *msg):
        self._parent_conn.send(msg)
        status, payload = self._parent_conn.recv()
        if status == "error":
            raise RuntimeError(f"Model subprocess error:\n{payload}")
        return payload

    def predict(self, image_rgb, instruction):
        return self._request("predict", image_rgb, instruction)

    def predict_from_obs(self, obs, instruction, image_key="agentview_image", wrist_image_key=None):
        return self._request("predict_from_obs", obs, instruction, image_key, wrist_image_key)

    def reset(self):
        self._request("reset")

    def close(self):
        if getattr(self, "_parent_conn", None) is not None:
            try:
                if self._proc.is_alive():
                    self._request("close")
            except Exception:
                pass
            self._parent_conn.close()
            self._parent_conn = None
        if getattr(self, "_proc", None) is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


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
        model_name: "random" | "openvla" | "openvla_oft" | "pi0"
        **kwargs: passed to the model constructor
                  e.g. device="cuda", unnorm_key="bridge_orig"
    """
    model_name = model_name.lower()
    isolate_process = kwargs.pop("isolate_process", False)
    if model_name == "random":
        return RandomModel(**kwargs)
    elif model_name == "openvla":
        if isolate_process:
            return SubprocessModel(OpenVLAModel, **kwargs)
        return OpenVLAModel(**kwargs)
    elif model_name in ("openvla_oft", "openvla-oft", "oft"):
        if isolate_process:
            return SubprocessModel(OpenVLAOFTModel, **kwargs)
        return OpenVLAOFTModel(**kwargs)
    elif model_name in ("pi0", "pi_zero"):
        return Pi0Model(**kwargs)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: random, openvla, openvla_oft, pi0"
        )
