import os

import pygame

from source.input import KeyboardInput


class Game:
    def __init__(self, state_dict, start_state, *, render_fps=60):
        self.screen = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        self.input = KeyboardInput()
        self.state_dict = state_dict
        self.state = self.state_dict[start_state]
        self.render_fps = render_fps
        self._accumulator = 0.0
        self.fixed_dt = 1.0 / 60.0

    def advance(self, elapsed_seconds):
        """Run 60 physics ticks/second; bound catch-up after a long stall."""
        self._accumulator += min(max(elapsed_seconds, 0.0), 0.25)
        ticks = 0
        while self._accumulator + 1e-12 >= self.fixed_dt:
            self.update()
            self.input.end_frame()
            self._accumulator = max(0.0, self._accumulator - self.fixed_dt)
            ticks += 1
        return ticks

    def update(self):
        self.state.update(self.screen, self.input)
        if self.state.finished:
            summarize = getattr(self.state, "result_summary", None)
            summary = summarize() if summarize is not None else None
            next_state = self.state.next
            self.state.finished = False
            leave = getattr(self.state, "leave", None)
            if leave is not None:
                leave()
            self.state = self.state_dict[next_state]
            enter = getattr(self.state, 'enter', None)
            if enter is not None:
                enter()
            receive = getattr(self.state, "receive_result", None)
            if receive is not None and summary is not None:
                receive(summary)
            draw = getattr(self.state, "draw", None)
            if draw is not None:
                draw(self.screen)

    def run(self, max_frames=None):
        """Run the legacy loop; ``max_frames`` supports automated smoke tests."""

        frame_count = 0
        running = True
        while running:
            elapsed = self.clock.tick(self.render_fps) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self.input.handle_event(event)

            if not running:
                break

            self.input.sync(pygame.key.get_pressed())
            self.advance(elapsed)
            pygame.display.update()
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                running = False

        pygame.quit()


def load_graphics(path, accept=('.jpg', '.png', '.bmp', '.gif')):
    graphics = {}
    for pic in os.listdir(path):
        name, ext = os.path.splitext(pic)
        if ext.lower() in accept:
            img = pygame.image.load(os.path.join(path, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
            graphics[name] = img
    return graphics


def get_image(sheet, x, y, width, height, colorkey, scale):
    image = pygame.Surface((width, height))
    image.blit(sheet, (0, 0), (x, y, width, height))
    image.set_colorkey(colorkey)
    image = pygame.transform.scale(image, (int(width * scale), int(height * scale)))
    return image
