import numpy as np
import pyglet
from env_global import *
from pyglet.gl import *



class Car(object):
    def __init__(self, width=1.65, length=4.2, color=(1, 0, 0)):
        self.width = width
        self.length = length
        self.color = color
        # half width, half length and in pixels metric
        self.hw = width / 2
        self.hl = length / 2
        self.hwp = self.hw * PIXELS_PER_METER
        self.hlp = self.hl * PIXELS_PER_METER


class PathEnv(object):
    viewer = None
    dt = .1  # refresh rate
    action_bound = np.array([5, 35], dtype=np.float32)
    # x, y, v, v_direction
    state_dim = 4
    # acceleration, a_direction
    action_dim = 2

    def __init__(self):
        # state information
        # including position info: (x, y)
        # and velocity info:(v, v_direction)
        self.state = np.zeros(
            1, dtype=[('x', np.float32),
                      ('y', np.float32),
                      ('v', np.float32),
                      ('v_d',np.float32)])
        self.state['x'] = ORIGIN[0]
        self.state['y'] = ORIGIN[1]
        self.state['v'] = 0
        self.state['v_d'] = 90
        self.car = Car()
        self.timer = 0
        self.last_a = None # last acceleration


    def step(self, action):
        self.timer += 1
        done = False
        # calc angles in rad
        v_dr = self.state['v_d'] * np.pi / 180
        a_dr = action[1] * np.pi / 180
        # calc components
        vx = self.state['v'] * np.cos(v_dr)
        vy = self.state['v'] * np.sin(v_dr)
        ax = action[0] * np.cos(a_dr)
        ay = action[1] * np.sin(a_dr)
        # calc x_, y_, vx_, vy_
        x_ = self.state['x'] + vx * self.dt + 0.5 * ax * self.dt ** 2
        y_ = self.state['y'] + vy * self.dt + 0.5 * ay * self.dt ** 2
        vx_ = vx + ax * self.dt
        vy_ = vy + ay * self.dt
        v_ = np.sqrt(vx_ ** 2 + vy_ ** 2)
        v_d_ = np.arctan2(vx_, vy_) * 180 / np.pi

        self.state['x'], self.state['y'] = x_, y_
        self.state['v'], self.state['v_d'] = v_, v_d_

        # state
        s = np.concatenate((x_, y_, v_, v_d_))

        # reward
        r = 0

        # out of range punish
        if not self._on_road():
            r -= out_of_range_loss
            print('out of range')
            done = True

        # distance
        distance = np.sqrt((self.state['x'] - DESTINATION[0]) ** 2 +
                           (self.state['y'] - DESTINATION[1]) ** 2)
        if distance < DESTINATION_RADIUS:
            done = True
        r -= distance * distance_coeff

        # speed restriction
        if v_ > SPEAD_LIMIT:
            r -= overspeed_loss
            done = True

        # drive direction
        r -= self.state['v_d'] * direction_coeff

        # jerk
        if self.last_a is not None:
            r -= (action[0] - self.last_a) / self.dt * jerk_coeff
        self.last_a = action[0]

        # acceleration
        r -= action[0] * acceleration_coeff

        # time consumption
        r -= self.timer * time_loss_coeff

        if done:
            self.timer = 0

        return s, r, done

    def reset(self):
        self.state['x'], self.state['y'] = ORIGIN
        self.state['v'], self.state['v_d'] = INITIAL_V
        self.timer = 0
        return np.concatenate((self.state['x'], self.state['y'], self.state['v'], self.state['v_d']))


    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.state, self.car)
        self.viewer.render()


    def _on_road(self):
        # judge if the car one the road, use RAD metric
        ans = True
        phi = np.arctan2(self.car.width, self.car.length)
        theta = [-phi, phi, -phi + np.pi, phi + np.pi]

        r = np.sqrt(self.car.hw ** 2 + self.car.hl ** 2)
        # judge
        for t in theta:
            t += self.state['v_d']
            y = self.state['y'] + r * np.sin(t)
            ans = ans and (y < ROAD_WIDTH) and (y > 0)
        return ans




class Viewer(pyglet.window.Window):

    def __init__(self, info, car):
        super(Viewer, self).__init__(width=int(WIDTH), height=int(HEIGHT),
                                    resizable=False,
                                    caption='VisualPath',
                                    vsync=False)
        # self.set_fullscreen(True)
        self.state = info
        self.car = car
        pyglet.gl.glClearColor(0.0, 0.93, 0.0, 1.0)
        self.batch = pyglet.graphics.Batch()

    def render(self):
        self._update_path()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def _drawOneLine(self, x1, y1, x2, y2):
        glBegin(GL_LINES)
        glVertex2f(x1, y1)
        glVertex2f(x2, y2)
        glEnd()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self._draw_OD()
        self._draw_road()
        self._draw_car()

    def _draw_OD(self):
        glColor3f(0.0, 0.93, 0.0)
        glPointSize(4)
        glBegin(GL_POINTS)
        glVertex2f(ORIGIN[0], ORIGIN[1])
        glVertex2f(DESTINATION[0], DESTINATION[1])
        glEnd()

    def _draw_road(self):
        # road
        glColor3f(0.50, 0.54, 0.53)
        glRectf(0, 0,
                WIDTH, ROAD_WIDTH * PIXELS_PER_METER)

        # white line
        glEnable(GL_LINE_STIPPLE)
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(7.5)
        glLineStipple(10, 0xFFF0)
        self._drawOneLine(0.0, HEIGHT / 2.0, WIDTH, HEIGHT / 2.0)
        glDisable(GL_LINE_STIPPLE)

        # yellow line
        glColor3f(1.0, 1.0, 0.0)
        glLineWidth(15)
        self._drawOneLine(0.0, HEIGHT, WIDTH, HEIGHT)


    def _draw_car(self):
        # draw car
        glPushMatrix()
        glTranslatef(self.state['x'] * PIXELS_PER_METER,
                     self.state['y'] * PIXELS_PER_METER, 0)
        glRotatef(self.state['v_d'], 0.0, 0.0, 1.0)
        glColor3f(self.car.color[0], self.car.color[1],
                  self.car.color[2])
        glRectf(-self.car.hlp, -self.car.hwp, self.car.hlp, self.car.hwp)
        glPopMatrix()

        glColor3f(1 - self.car.color[0],1 - self.car.color[1],
                  1 - self.car.color[2])
        glPointSize(4)
        glBegin(GL_POINTS)
        glVertex2f(self.state['x'] * PIXELS_PER_METER,
                   self.state['y'] * PIXELS_PER_METER)
        glEnd()


    def _update_path(self):
        pass




if __name__ == '__main__':
    print(screen.width, screen.height)
    env = PathEnv()
    while True:
        env.render()



