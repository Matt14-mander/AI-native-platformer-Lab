"""State v2 uses fixed-height jumps and faster movement response; 15 features."""

from ai_platformer.settings import AI_GAMEPLAY_PATH, load_gameplay_settings

from .platformer_state_v1 import PlatformerStateEnvV1


class PlatformerStateEnvV2(PlatformerStateEnvV1):
    def __init__(self, **kwargs):
        if kwargs.get("settings") is None:
            kwargs["settings"] = load_gameplay_settings(AI_GAMEPLAY_PATH)
        super().__init__(**kwargs)
