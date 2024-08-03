import pygame as pg
import numpy as np
import pygame
import time



def draw_grid(scr, line_color, bg_color, WIDTH=800, HEIGHT=800, Col_num=16, Row_num=16):
    '''a function to draw gridlines and other objects'''
    # Horizontal lines
    for j in range(Row_num + 1):
        pg.draw.line(scr, line_color, (0, j * HEIGHT // Row_num), (WIDTH, j * HEIGHT // Row_num), 2)
    # # Vertical lines
    for i in range(Col_num + 1):
        pg.draw.line(scr, line_color, (i * WIDTH // Col_num, 0), (i * WIDTH // Col_num, HEIGHT), 2)

    for x1 in range(0, WIDTH, WIDTH // Col_num):
        for y1 in range(0, HEIGHT, HEIGHT // Row_num):
            rect = pg.Rect(x1, y1, WIDTH // Col_num, HEIGHT // Row_num)
            pg.draw.rect(scr, bg_color, rect, 1)


def animate(rescue_team_traj, victims_traj, rescue_team_vfd, rescue_team_vfd_status, rescue_team_roles, env_map,
            WIDTH=800, HEIGHT=800, Col_num=16, Row_num=16, wait_time=0.0):
    pygame.init()
    # define colors
    bg_color = pg.Color(255, 255, 255)
    line_color = pg.Color(128, 128, 128)
    vfdr_color = pg.Color(8, 136, 8, 128)
    vfds_color = pg.Color(255, 165, 0, 128)
    vfdrs_color = pg.Color(173, 216, 230, 128)
    font = pg.font.SysFont('arial', 20)

    num_rescue_team = len(rescue_team_traj)
    num_victims = len(victims_traj)

    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH + 2, HEIGHT + 2))  # set up the screen
    pg.display.set_caption("gridworld")  # add a caption
    bg = pg.Surface(screen.get_size())  # get a background surface
    bg = bg.convert()

    img_rescuer = pg.image.load('TurtleBot.png')
    img_mdf_r = pg.transform.scale(img_rescuer, (WIDTH // Col_num, HEIGHT // Row_num))

    img_rescuer_scout = pg.image.load('typhoon.jpg')
    img_mdf_rs = pg.transform.scale(img_rescuer_scout, (WIDTH // Col_num, HEIGHT // Row_num))

    img_scout = pg.image.load('Crazyflie.JPG')
    img_mdf_s = pg.transform.scale(img_scout, (WIDTH // Col_num, HEIGHT // Row_num))

    img_victim = pg.image.load('victim.png')
    img_mdf_victim = pg.transform.scale(img_victim, (WIDTH // Col_num, HEIGHT // Row_num))

    img_wall = pg.image.load('wall.png')
    img_mdf_wall = pg.transform.scale(img_wall, (WIDTH // Col_num, HEIGHT // Row_num))

    bg.fill(bg_color)
    screen.blit(bg, (0, 0))
    clock = pg.time.Clock()
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
        step = -1
        list_victims = np.arange(num_victims).tolist()
        list_rescue_team = np.arange(num_rescue_team).tolist()

        for rescue_team_stt, victims_stt in zip(np.moveaxis(rescue_team_traj, 0, -1),
                                                np.moveaxis(victims_traj, 0, -1)):

            for row in range(Row_num):
                for col in range(Col_num):
                    if env_map[row, col] == 1:
                        screen.blit(img_mdf_wall,
                                    (col * (WIDTH // Col_num),
                                     row * (HEIGHT // Row_num)))
            step += 1
            for num in list_rescue_team:
                if str(rescue_team_roles[num]) == "b'rs'":
                    vfd_color = vfdrs_color
                elif str(rescue_team_roles[num]) == "b'r'":
                    vfd_color = vfdr_color
                elif str(rescue_team_roles[num]) == "b's'":
                    vfd_color = vfds_color

                # rescuer visual field depth
                vfd_j = 0
                for j in range(int(max(rescue_team_stt[1, num] - rescue_team_vfd[num], 0)),
                               int(min(Col_num, rescue_team_stt[1, num] + rescue_team_vfd[num] + 1))):
                    vfd_i = 0
                    for i in range(int(max(rescue_team_stt[0, num] - rescue_team_vfd[num], 0)),
                                   int(min(Row_num, rescue_team_stt[0, num] + rescue_team_vfd[num] + 1))):
                        if rescue_team_vfd_status[num][step][vfd_i, vfd_j]:
                            rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                           (WIDTH // Col_num), (HEIGHT // Row_num))
                            pg.draw.rect(screen, vfd_color, rect)
                        vfd_i += 1
                    vfd_j += 1

            # agents
            for num in list_rescue_team:
                if str(rescue_team_roles[num]) == "b'rs'":
                    img_mdf = img_mdf_rs
                elif str(rescue_team_roles[num]) == "b'r'":
                    img_mdf = img_mdf_r
                elif str(rescue_team_roles[num]) == "b's'":
                    img_mdf = img_mdf_s
                screen.blit(img_mdf,
                            (rescue_team_stt[1, num] * (WIDTH // Col_num),
                             rescue_team_stt[0, num] * (HEIGHT // Row_num)))
                screen.blit(font.render(str(num + 1), True, (0, 0, 0)),
                            (rescue_team_stt[1, num] * (WIDTH // Col_num),
                             rescue_team_stt[0, num] * (HEIGHT // Row_num)))

                # Stop showing finished agents
                # if (step >= 1 and
                #     (rescue_team_stt[:, num][0] == rescue_team_history[:, num][0] == rescue_team_traj[num, -1, 0] and
                #      rescue_team_stt[:, num][1] == rescue_team_history[:, num][1] == rescue_team_traj[num, -1, 1])):
                #     list_rescue_team.remove(num)

            for num in list_victims:
                screen.blit(img_mdf_victim, (victims_stt[1, num] * (WIDTH // Col_num),
                                             victims_stt[0, num] * (HEIGHT // Row_num)))
                screen.blit(font.render(str(num + 1), True, (0, 0, 0)),
                            (victims_stt[1, num] * (WIDTH // Col_num), victims_stt[0, num] * (HEIGHT // Row_num)))

                # Stop showing rescued victims
                # if step >= 1 and (victims_stt[:, num][0] == victims_history[:, num][0] == victims_traj[num, -1, 0] and
                #                   victims_stt[:, num][1] == victims_history[:, num][1] == victims_traj[num, -1, 1]):
                #     list_victims.remove(num)

            draw_grid(screen, line_color, bg_color, WIDTH, HEIGHT, Col_num, Row_num)
            pg.display.flip()
            pg.display.update()
            time.sleep(wait_time)  # wait between the shows

            for num in list_victims:
                screen.blit(bg, (victims_stt[1, num] * (WIDTH // Col_num),
                                 victims_stt[0, num] * (HEIGHT // Row_num)))

            for num in list_rescue_team:
                screen.blit(bg, (rescue_team_stt[1, num] * (WIDTH // Col_num),
                                 rescue_team_stt[0, num] * (HEIGHT // Row_num)))

                # rescuer visual field depths
                for j in range(int(max(rescue_team_stt[1, num] - rescue_team_vfd[num], 0)),
                               int(min(Row_num, rescue_team_stt[1, num] + rescue_team_vfd[num] + 1))):
                    for i in range(int(max(rescue_team_stt[0, num] - rescue_team_vfd[num], 0)),
                                   int(min(Col_num, rescue_team_stt[0, num] + rescue_team_vfd[num] + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, bg_color, rect)

            victims_history = victims_stt
            rescue_team_history = rescue_team_stt

        for num in list_rescue_team:
            if str(rescue_team_roles[num]) == "b'rs'":
                img_mdf = img_mdf_rs
            elif str(rescue_team_roles[num]) == "b'r'":
                img_mdf = img_mdf_r
            elif str(rescue_team_roles[num]) == "b's'":
                img_mdf = img_mdf_s
            screen.blit(img_mdf, (rescue_team_traj[num, -1, 1] * (WIDTH // Col_num),
                                  rescue_team_traj[num, -1, 0] * (HEIGHT // Row_num)))
        for num in list_victims:
            screen.blit(img_mdf_victim, (victims_traj[num, -1, 1] * (WIDTH // Col_num),
                                         victims_traj[num, -1, 0] * (HEIGHT // Row_num)))

        draw_grid(screen, line_color, bg_color, WIDTH, HEIGHT, Col_num, Row_num)
        pg.display.flip()
        pg.display.update()
        run = False
    pg.quit()


class training_animate:
    def __init__(self, num_rescue_team, num_victims, WIDTH=800, HEIGHT=800, Col_num=16, Row_num=16):
        pygame.init()
        self.width = WIDTH
        self.height = HEIGHT
        self.number_of_columns = Col_num
        self.number_of_rows = Row_num
        self.bg_color = pg.Color(255, 255, 255)
        self.line_color = pg.Color(128, 128, 128)
        self.vfdr_color = pg.Color(8, 136, 8, 128)
        self.vfds_color = pg.Color(255, 165, 0, 128)
        self.vfdrs_color = pg.Color(173, 216, 230, 128)
        self.font = pg.font.SysFont('arial', 20)

        self.num_rescue_team = num_rescue_team
        self.num_victims = num_victims
        self.list_victims = np.arange(self.num_victims).tolist()
        self.list_rescue_team = np.arange(self.num_rescue_team).tolist()
        pg.init()  # initialize pygame
        self.screen = pg.display.set_mode((WIDTH + 2, HEIGHT + 2))  # set up the screen
        pg.display.set_caption("gridworld")  # add a caption
        self.bg = pg.Surface(self.screen.get_size())  # get a background surface
        self.bg = self.bg.convert()

        self.img_rescuer = pg.image.load('TurtleBot.png')
        self.img_mdf_r = pg.transform.scale(self.img_rescuer, (WIDTH // Col_num, HEIGHT // Row_num))

        self.img_rescuer_scout = pg.image.load('typhoon.jpg')
        self.img_mdf_rs = pg.transform.scale(self.img_rescuer_scout, (WIDTH // Col_num, HEIGHT // Row_num))

        self.img_scout = pg.image.load('Crazyflie.JPG')
        self.img_mdf_s = pg.transform.scale(self.img_scout, (WIDTH // Col_num, HEIGHT // Row_num))

        self.img_victim = pg.image.load('victim.png')
        self.img_mdf_victim = pg.transform.scale(self.img_victim, (WIDTH // Col_num, HEIGHT // Row_num))

        self.img_wall = pg.image.load('wall.png')
        self.img_mdf_wall = pg.transform.scale(self.img_wall, (WIDTH // Col_num, HEIGHT // Row_num))

        self.bg.fill(self.bg_color)
        self.screen.blit(self.bg, (0, 0))
        self.clock = pg.time.Clock()
        pg.display.flip()

    def animate(self, rescue_team_traj, victims_traj, rescue_team_vfd, rescue_team_vfd_status, rescue_team_roles,
                rescue_team_sensation, rescue_team_dist2home, rescue_team_tasks, env_map, wait_time):
        self.num_victims = len(victims_traj)
        # run = True
        # while run:
        self.clock.tick(60)
        # for event in self.pg.event.get():
        #     if event.type == self.pg.QUIT:
        #         run = False

        for rescue_team_stt, victims_stt in zip(np.moveaxis(rescue_team_traj, 0, -1),
                                                np.moveaxis(victims_traj, 0, -1)):

            for row in range(self.number_of_rows):
                for col in range(self.number_of_columns):
                    if env_map[row, col] == 1:
                        self.screen.blit(self.img_mdf_wall,
                                    (col * (self.width // self.number_of_columns),
                                     row * (self.height // self.number_of_rows)))
            for num in self.list_rescue_team:
                if str(rescue_team_roles[num]) == 'rs':
                    vfd_color = self.vfdrs_color
                elif str(rescue_team_roles[num]) == 'r':
                    vfd_color = self.vfdr_color
                elif str(rescue_team_roles[num]) == 's':
                    vfd_color = self.vfds_color

                # rescuer visual field depth
                vfd_j = 0
                for j in range(int(max(rescue_team_stt[1, num] - rescue_team_vfd[num], 0)),
                               int(min(self.number_of_columns, rescue_team_stt[1, num] + rescue_team_vfd[num] + 1))):
                    vfd_i = 0
                    for i in range(int(max(rescue_team_stt[0, num] - rescue_team_vfd[num], 0)),
                                   int(min(self.number_of_rows, rescue_team_stt[0, num] + rescue_team_vfd[num] + 1))):
                        if rescue_team_vfd_status[num][vfd_i, vfd_j]:
                            rect = pg.Rect(j * (self.width // self.number_of_columns), i * (self.height // self.number_of_rows),
                                           (self.width // self.number_of_columns), (self.height // self.number_of_rows))

                            pg.draw.rect(self.screen, vfd_color, rect)
                            self.screen.blit(self.font.render(str(rescue_team_sensation[num]), True, (0, 0, 0)),
                                             (j * (self.width // self.number_of_columns), i * (self.height // self.number_of_rows)))
                            self.screen.blit(self.font.render('R'+str(rescue_team_dist2home[num]), True, (0, 0, 0)),
                                             (j * (self.width // self.number_of_columns)+20,
                                              i * (self.height // self.number_of_rows)+20))
                        vfd_i += 1
                    vfd_j += 1

            # agents
            for num in self.list_rescue_team:
                if str(rescue_team_roles[num]) == 'rs':
                    img_mdf = self.img_mdf_rs
                elif str(rescue_team_roles[num]) == 'r':
                    img_mdf = self.img_mdf_r
                elif str(rescue_team_roles[num]) == 's':
                    img_mdf = self.img_mdf_s
                self.screen.blit(img_mdf,
                            (rescue_team_stt[1, num] * (self.width // self.number_of_columns),
                             rescue_team_stt[0, num] * (self.height // self.number_of_rows)))
                self.screen.blit(self.font.render(str(num), True, (0, 0, 0)),
                            (rescue_team_stt[1, num] * (self.width // self.number_of_columns),
                             rescue_team_stt[0, num] * (self.height // self.number_of_rows)))
                self.screen.blit(self.font.render(rescue_team_tasks[num], True, (0, 0, 0)),
                                 (rescue_team_stt[1, num] * (self.width // self.number_of_columns)+30,
                                  rescue_team_stt[0, num] * (self.height // self.number_of_rows)+30))
                # Stop showing finished agents
                # if (step >= 1 and
                #     (rescue_team_stt[:, num][0] == rescue_team_history[:, num][0] == rescue_team_traj[num, -1, 0] and
                #      rescue_team_stt[:, num][1] == rescue_team_history[:, num][1] == rescue_team_traj[num, -1, 1])):
                #     list_rescue_team.remove(num)

            for num in self.list_victims:
                self.screen.blit(self.img_mdf_victim, (victims_stt[1, num] * (self.width // self.number_of_columns),
                                             victims_stt[0, num] * (self.height // self.number_of_rows)))
                self.screen.blit(self.font.render(str(num), True, (0, 0, 0)),
                            (victims_stt[1, num] * (self.width // self.number_of_columns), victims_stt[0, num] * (self.height // self.number_of_rows)))

                # Stop showing rescued victims
                # if step >= 1 and (victims_stt[:, num][0] == victims_history[:, num][0] == victims_traj[num, -1, 0] and
                #                   victims_stt[:, num][1] == victims_history[:, num][1] == victims_traj[num, -1, 1]):
                #     list_victims.remove(num)

            draw_grid(self.screen, self.line_color, self.bg_color, self.width, self.height,
                      self.number_of_columns, self.number_of_rows)
            pg.display.flip()
            pg.display.update()
            time.sleep(wait_time)  # wait between the shows

            for num in self.list_victims:
                self.screen.blit(self.bg, (victims_stt[1, num] * (self.width // self.number_of_columns),
                                 victims_stt[0, num] * (self.height // self.number_of_rows)))

            for num in self.list_rescue_team:
                self.screen.blit(self.bg, (rescue_team_stt[1, num] * (self.width // self.number_of_columns),
                                 rescue_team_stt[0, num] * (self.height // self.number_of_rows)))

                # rescuer visual field depths
                for j in range(int(max(rescue_team_stt[1, num] - rescue_team_vfd[num], 0)),
                               int(min(self.number_of_rows, rescue_team_stt[1, num] + rescue_team_vfd[num] + 1))):
                    for i in range(int(max(rescue_team_stt[0, num] - rescue_team_vfd[num], 0)),
                                   int(min(self.number_of_columns, rescue_team_stt[0, num] + rescue_team_vfd[num] + 1))):
                        rect = pg.Rect(j * (self.width // self.number_of_columns), i * (self.height // self.number_of_rows),
                                       (self.width // self.number_of_columns), (self.height // self.number_of_rows))
                        pg.draw.rect(self.screen, self.bg_color, rect)

            victims_history = victims_stt
            rescue_team_history = rescue_team_stt

        for num in self.list_rescue_team:
            if str(rescue_team_roles[num]) == "b'rs'":
                img_mdf = self.img_mdf_rs
            elif str(rescue_team_roles[num]) == "b'r'":
                img_mdf = self.img_mdf_r
            elif str(rescue_team_roles[num]) == "b's'":
                img_mdf = self.img_mdf_s
            self.screen.blit(img_mdf, (rescue_team_traj[num][0][1] * (self.width // self.number_of_columns),
                                  rescue_team_traj[num][0][0] * (self.height // self.number_of_rows)))

            # draw_grid(self.screen)
            # pg.display.flip()
            # pg.display.update()
            # run = False

        # pg.quit()