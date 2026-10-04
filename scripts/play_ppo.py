"""Play SB3 checkpoints or TinyInfer deployment actors in the shared renderer."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="model.zip (SB3) or actor.onnx (TinyInfer), with sidecar",
    )
    parser.add_argument("--backend", choices=("sb3", "tinyinfer"), default="sb3")
    parser.add_argument("--library", type=Path, help="TinyInfer bridge dynamic library")
    parser.add_argument(
        "--fuse-relu", action="store_true", help="enable TinyInfer Gemm/ReLU fusion"
    )
    parser.add_argument(
        "--task", choices=("flat", "obstacle", "gap", "mixed", "full"), default="gap"
    )
    parser.add_argument("--suite", choices=("train", "validation", "ood"), default="validation")
    parser.add_argument("--level-id", help="play one explicit level instead of a task suite")
    parser.add_argument("--seed", type=int, help="default: first checkpoint validation seed")
    parser.add_argument("--sampled", action="store_true", help="sample actions instead of argmax")
    parser.add_argument("--speed", type=float, default=1.0, help="wall-clock speed, 0.1 to 8")
    parser.add_argument(
        "--episodes", type=int, default=0, help="completed episodes; 0 cycles forever"
    )
    parser.add_argument(
        "--headless", action="store_true", help="render offscreen, run without delays"
    )
    parser.add_argument("--screenshot", type=Path, help="save the final rendered frame as PNG")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0.1 <= args.speed <= 8:
        parser.error("speed must be between 0.1 and 8")
    if args.episodes < 0 or (args.headless and args.episodes == 0):
        parser.error("episodes must be nonnegative; headless playback requires --episodes > 0")
    if args.seed is not None and args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.backend == "tinyinfer" and args.sampled:
        parser.error("TinyInfer playback supports deterministic actions; remove --sampled")
    if args.screenshot and args.screenshot.suffix.lower() != ".png":
        parser.error("screenshot must have a .png suffix")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    import pygame

    from ai_platformer.deployment.backends import load_backend
    from ai_platformer.rendering.ppo_playback import PlaybackSession, create_renderer

    loaded = None
    try:
        loaded = load_backend(
            args.backend, args.model, library=args.library, fuse_relu=args.fuse_relu
        )
        factory = loaded.factory
        policy = loaded.policy
        path = loaded.model_path
        levels = (
            [args.level_id]
            if args.level_id
            else [factory.level_id]
            if args.task == "full" and "full" not in factory.repository.manifest["splits"]
            else factory.repository.split(args.task, args.suite)
        )
        for level in levels:
            factory.repository.load(level)
        seed = args.seed if args.seed is not None else loaded.default_seed
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as error:
        if loaded is not None:
            loaded.close()
        parser.error(str(error))

    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((1000, 600))
        pygame.display.set_caption(f"{loaded.label} playback | {path.name}")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()
    except BaseException:
        loaded.close()
        pygame.quit()
        raise
    session = None
    index = completed = 0
    speed = args.speed
    paused = False
    accumulator = terminal_elapsed = 0.0
    running = True

    def start_level():
        nonlocal session, accumulator, terminal_elapsed, paused
        if session is not None:
            session.env.close()
        session = PlaybackSession(
            policy, factory, level_id=levels[index], seed=seed, deterministic=not args.sampled
        )
        accumulator = terminal_elapsed = 0.0
        paused = False
        return create_renderer(factory, session, screen.get_size())

    try:
        renderer = start_level()
        print("P/Space: pause | R: replay | N: next level | [/]: speed | Esc: quit")
        while running:
            elapsed = clock.tick(0 if args.headless else factory.settings.render_fps) / 1000
            reset_clock = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_p, pygame.K_SPACE):
                        paused = not paused
                        reset_clock = True
                    elif event.key in (pygame.K_r, pygame.K_n):
                        if event.key == pygame.K_n:
                            index = (index + 1) % len(levels)
                        renderer = start_level()
                        reset_clock = True
                    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        speed = max(
                            0.1,
                            min(8.0, speed * (0.5 if event.key == pygame.K_LEFTBRACKET else 2.0)),
                        )
                        reset_clock = True
            if not running:
                break
            if reset_clock:
                accumulator = 0.0
                elapsed = 0.0
            if not paused and not session.done:
                # A decision still advances exactly action_repeat core ticks.
                interval = factory.action_repeat / factory.settings.render_fps / speed
                accumulator += interval if args.headless else min(elapsed, 0.25)
                while accumulator >= interval and not session.done:
                    session.step()
                    accumulator -= interval
                    if session.done:
                        completed += 1
                        print(
                            json.dumps(
                                {
                                    "backend": args.backend,
                                    "fuse_relu": args.fuse_relu,
                                    **session.summary(),
                                }
                            ),
                            flush=True,
                        )
                        terminal_elapsed = 0.0
            renderer.draw(screen, session.env.core.state)
            lines = [
                (
                    f"{loaded.label} | {session.env.level_id} | seed={seed} | "
                    f"{'sampled' if args.sampled else 'deterministic'} | {speed:g}x"
                ),
                (
                    f"step={session.info['episode_step']} | progress={session.info['progress']:.1%} | "
                    f"action={session.action.name} | return={session.episode_return:.2f} | "
                    f"infer={session.inference_ms:.2f} ms"
                ),
                (
                    f"outcome={session.info.get('outcome', 'playing')} | "
                    f"coins={session.info['coins_collected']}/{session.info['coins_total']} | "
                    "P/Space pause  R replay  N next  [/] speed  Esc quit"
                ),
            ]
            panel = pygame.Surface((screen.get_width(), 82), pygame.SRCALPHA)
            panel.fill((12, 20, 25, 210))
            screen.blit(panel, (0, 0))
            for line_index, line in enumerate(lines):
                screen.blit(font.render(line, True, (255, 240, 211)), (10, 8 + 24 * line_index))
            if paused:
                renderer.draw_pause(screen)
            pygame.display.flip()
            if session.done and not paused:
                terminal_elapsed += elapsed
                if args.headless or terminal_elapsed >= factory.settings.terminal_display_ms / 1000:
                    if args.episodes and completed >= args.episodes:
                        running = False
                    else:
                        index = (index + 1) % len(levels)
                        renderer = start_level()
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(args.screenshot))
    finally:
        if session is not None:
            session.env.close()
        loaded.close()
        pygame.quit()


if __name__ == "__main__":
    main()
