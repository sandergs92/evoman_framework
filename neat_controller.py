from controller import Controller

# implements controller structure for player
class player_controller(Controller):
    def control(self, inputs, controller):
        output = controller.activate(inputs)
        # takes decisions about sprite actions
        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0
        return [left, right, jump, shoot, release]

# implements controller structure for enemy
class enemy_controller(Controller):
    def control(self, inputs,controller):
        output = controller.activate(inputs)
        # takes decisions about sprite actions
        if output[0] > 0.5:
            attack1 = 1
        else:
            attack1 = 0

        if output[1] > 0.5:
            attack2 = 1
        else:
            attack2 = 0

        if output[2] > 0.5:
            attack3 = 1
        else:
            attack3 = 0
            
        if output[3] > 0.5:
            attack4 = 1
        else:
            attack4 = 0
        return [attack1, attack2, attack3, attack4]