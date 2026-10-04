"""Play a checked LevelSpec with keyboard, scripted controls or verified replay."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from ai_platformer.envs.factory import EnvironmentFactory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--controller", choices=("keyboard", "rule", "replay"), default="keyboard")
    parser.add_argument("--route", type=Path)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if (
        args.seed < 0
        or args.episodes < 0
        or not math.isfinite(args.speed)
        or not 0.1 <= args.speed <= 8
    ):
        parser.error("invalid seed, episode count or speed")
    if args.headless and (args.controller == "keyboard" or not args.episodes):
        parser.error("headless needs rule/replay controller and --episodes > 0")
    if (args.controller == "replay") != (args.route is not None):
        parser.error("--route is required only for replay controller")
    if args.screenshot and args.screenshot.suffix.lower() != ".png":
        parser.error("screenshot must be PNG")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = os.environ["SDL_AUDIODRIVER"] = "dummy"
    try:
        witness = None
        if args.route:
            from ai_platformer.content.routes import replay_route
            from scripts.replay_level_route import load_witness

            witness = load_witness(args.route)
            proof = replay_route(args.input, witness)
            if not proof["passed"]:
                raise ValueError("route replay mismatch; playback rejected")
            factory = EnvironmentFactory({**witness["environment"], "level_spec": str(args.input)})
            args.seed = witness["seed"]
        else:
            factory = EnvironmentFactory(
                {
                    "environment_id": "PlatformerState-v2",
                    "level_spec": str(args.input),
                    "action_repeat": 4,
                    "episode_step_limit": 768,
                }
            )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    from types import SimpleNamespace

    import pygame

    from ai_platformer.agents.scripted import RuleJumpAgent
    from ai_platformer.content.routes import transition_digest
    from ai_platformer.rendering.keyboard import PygameKeyboardController
    from ai_platformer.rendering.ppo_playback import create_renderer

    env = factory.make()
    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((1000, 600))
        pygame.display.set_caption(f"LevelSpec | {factory.level_id} | {args.controller}")
        renderer = create_renderer(factory, SimpleNamespace(env=env), screen.get_size())
        keyboard, agent = PygameKeyboardController(), RuleJumpAgent()
        font, clock = pygame.font.Font(None, 24), pygame.time.Clock()
        running, paused, done = True, False, False
        completed, index = 0, 0
        accumulator = terminal_elapsed = total = 0.0
        obs, info = env.reset(seed=args.seed)
        if witness and transition_digest(env, obs, info) != witness["initial_sha256"]:
            raise ValueError("playback initial state differs from route witness")
        agent.reset(seed=args.seed)
        while running:
            elapsed = clock.tick(0 if args.headless else factory.settings.render_fps) / 1000
            reset = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    paused = not paused
                    accumulator = 0.0
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    reset = True
            if not running:
                break
            if reset:
                obs, info = env.reset(seed=args.seed)
                agent.reset(seed=args.seed)
                renderer.reset()
                index = 0
                done = False
                total = accumulator = terminal_elapsed = 0.0
            interval = factory.action_repeat / factory.settings.render_fps / args.speed
            if not paused and not done:
                accumulator += interval if args.headless else min(elapsed, 0.25)
                while accumulator >= interval and not done:
                    action = (
                        int(keyboard.action(pygame.key.get_pressed()))
                        if args.controller == "keyboard"
                        else int(agent.act(obs))
                        if args.controller == "rule"
                        else witness["actions"][index]
                    )
                    obs, reward, term, trunc, info = env.step(action)
                    if (
                        witness
                        and transition_digest(env, obs, info, action=action, reward=reward)
                        != witness["step_sha256"][index]
                    ):
                        raise ValueError(f"playback route differs at step {index}")
                    index += 1
                    total += reward
                    done = term or trunc
                    accumulator -= interval
                    if done:
                        completed += 1
                        print(
                            json.dumps(
                                {
                                    "controller": args.controller,
                                    "level_id": factory.level_id,
                                    "outcome": info["outcome"],
                                    "steps": info["episode_step"],
                                    "ticks": info["tick"],
                                    "coins_collected": info["coins_collected"],
                                    "coins_total": info["coins_total"],
                                    "return": total,
                                }
                            ),
                            flush=True,
                        )
            renderer.draw(screen, env.core.state)
            panel = pygame.Surface((1000, 58), pygame.SRCALPHA)
            panel.fill((12, 20, 25, 210))
            screen.blit(panel, (0, 0))
            text = f"{args.controller} | {factory.level_id} | step={index} | {info.get('outcome', 'playing')}"
            screen.blit(font.render(text, True, (255, 240, 211)), (10, 8))
            screen.blit(
                font.render(
                    f"coins={info['coins_collected']}/{info['coins_total']} | P pause  R restart  Esc quit",
                    True,
                    (255, 240, 211),
                ),
                (10, 32),
            )
            if paused:
                renderer.draw_pause(screen)
            pygame.display.flip()
            if done and not paused:
                terminal_elapsed += elapsed
                if args.headless or terminal_elapsed >= factory.settings.terminal_display_ms / 1000:
                    if args.episodes and completed >= args.episodes:
                        running = False
                    else:
                        obs, info = env.reset(seed=args.seed)
                        agent.reset(seed=args.seed)
                        renderer.reset()
                        index = 0
                        done = False
                        total = accumulator = terminal_elapsed = 0.0
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(args.screenshot))
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
