import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vector import Vec3


class SparkyV2(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        
        # Game state tracking
        self.is_kickoff = False
        self.on_defense = False
        self.dribbling = False
        self.going_for_aerial = False
        self.is_in_position = False
        
        # Constants and thresholds
        self.AERIAL_HEIGHT_THRESHOLD = 300  # Minimum ball height to consider aerial
        self.DRIBBLE_DISTANCE = 100  # Distance for dribbling
        self.ROTATION_DISTANCE = 2000  # Distance to consider rotating
        self.LOW_BOOST_THRESHOLD = 30  # Threshold to consider going for boost
        self.DEFENSE_THRESHOLD = 3000  # Distance from goal to trigger defense mode

    def initialize_agent(self):
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Update game information
        self.boost_pad_tracker.update_boost_status(packet)
        
        # Initialize controller state
        controller = SimpleControllerState()
        
        # Get car and ball information
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        car_rotation = my_car.physics.rotation
        car_angular_velocity = Vec3(my_car.physics.angular_velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)
        
        # Get team goal information
        team_sign = 1 if self.team == 0 else -1
        enemy_goal = Vec3(0, team_sign * 5120, 0)
        own_goal = Vec3(0, -team_sign * 5120, 0)
        
        # Check if we're executing a sequence
        if self.active_sequence is not None and not self.active_sequence.done:
            return self.active_sequence.tick(packet)
        
        # Determine if kickoff is happening
        kickoff = packet.game_info.is_kickoff_pause
        if kickoff:
            return self.handle_kickoff(packet, car_location, ball_location)
        
        # Get boost if low and nearby boost is available
        if my_car.boost < self.LOW_BOOST_THRESHOLD and not self.is_critical_situation(car_location, ball_location, own_goal):
            nearest_boost = self.find_nearest_boost(car_location)
            if nearest_boost is not None:
                controller = self.go_to_target(car_location, car_velocity, nearest_boost)
                return controller
        
        # Check defensive situation
        if self.should_defend(car_location, ball_location, own_goal):
            return self.handle_defense(car_location, car_velocity, ball_location, ball_velocity, own_goal)
        
        # Check for aerial opportunity
        if self.should_aerial(ball_location, ball_velocity):
            return self.handle_aerial(car_location, car_velocity, ball_location, ball_velocity, enemy_goal)
        
        # Check for dribbling opportunity
        if self.should_dribble(car_location, ball_location, ball_velocity):
            return self.handle_dribble(car_location, car_velocity, ball_location, ball_velocity, enemy_goal)
            
        # Handle regular shooting
        return self.handle_shooting(car_location, car_velocity, ball_location, ball_velocity, enemy_goal)

    def handle_kickoff(self, packet, car_location, ball_location):
        """Handle kickoff with speedflip if appropriate"""
        controller = SimpleControllerState()
        controller.boost = True
        
        # Calculate angle to ball
        angle_to_ball = math.atan2(ball_location.y - car_location.y, ball_location.x - car_location.x)
        controller.steer = steer_toward_target(self.get_car_facing_vector(packet), ball_location - car_location)
        
        # If we're aligned, perform speedflip sequence
        if abs(controller.steer) < 0.1 and (ball_location - car_location).length() < 3000:
            self.active_sequence = self.speedflip_sequence()
            return self.active_sequence.tick(packet)
        
        controller.throttle = 1.0
        return controller

    def speedflip_sequence(self):
        """Return a sequence for performing a speedflip"""
        sequence = Sequence([
            ControlStep(duration=0.1, controls=SimpleControllerState(throttle=1, boost=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle=1, boost=True, jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle=1, boost=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle=1, boost=True, jump=True, pitch=-1)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle=1, boost=True, pitch=-1)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle=1, boost=True, pitch=-1, roll=1, yaw=1)),
            ControlStep(duration=0.3, controls=SimpleControllerState(throttle=1, boost=True, pitch=-1, roll=1, yaw=1)),
            ControlStep(duration=0.1, controls=SimpleControllerState(throttle=1, boost=True, pitch=1, roll=-1, yaw=-1)),
            ControlStep(duration=0.2, controls=SimpleControllerState(throttle=1, boost=True)),
        ])
        return sequence

    def should_defend(self, car_location, ball_location, own_goal):
        """Determine if we should focus on defense"""
        ball_to_goal = (own_goal - ball_location).length()
        car_to_goal = (own_goal - car_location).length()
        
        # Defense is needed if ball is in our half and close to goal
        return ball_location.y * (1 if self.team == 0 else -1) < 0 and ball_to_goal < self.DEFENSE_THRESHOLD

    def handle_defense(self, car_location, car_velocity, ball_location, ball_velocity, own_goal):
        """Defensive positioning and clears"""
        controller = SimpleControllerState()
        
        # Predict where the ball will be 
        future_ball_location = ball_location + ball_velocity.scaled(0.5)
        
        # Position between ball and goal
        target_position = self.get_defensive_position(ball_location, future_ball_location, own_goal)
        
        # If we're in position and ball is coming toward goal, intercept
        if (target_position - car_location).length() < 200:
            # Aim toward enemy half for the clear
            direction = ball_location - own_goal
            target = ball_location + direction.normalized().scaled(100)
            controller = self.go_to_target(car_location, car_velocity, target)
            
            # Jump/flip if close to ball
            if (ball_location - car_location).length() < 200:
                controller.jump = True
                if abs(car_velocity.length()) > 800:
                    controller.pitch = -1 if car_location.y * (1 if self.team == 0 else -1) > 0 else 1
        else:
            # Get in position
            controller = self.go_to_target(car_location, car_velocity, target_position)
            
        return controller

    def get_defensive_position(self, ball_location, future_ball_location, own_goal):
        """Calculate optimal defensive position"""
        # Get vector from goal to ball
        goal_to_ball = ball_location - own_goal
        
        # Get point between goal and ball, closer to goal
        return own_goal + goal_to_ball.normalized().scaled(goal_to_ball.length() * 0.6)

    def should_aerial(self, ball_location, ball_velocity):
        """Determine if we should go for an aerial"""
        return ball_location.z > self.AERIAL_HEIGHT_THRESHOLD and ball_velocity.z > 0

    def handle_aerial(self, car_location, car_velocity, ball_location, ball_velocity, enemy_goal):
        """Handle aerial shots"""
        controller = SimpleControllerState()
        
        # Predict ball path
        time_to_reach = self.predict_time_to_reach(car_location, car_velocity, ball_location)
        future_ball_location = ball_location + ball_velocity.scaled(time_to_reach)
        
        # If ball is going to be high, position for aerial
        if future_ball_location.z > self.AERIAL_HEIGHT_THRESHOLD:
            # Position underneath the ball
            aerial_position = Vec3(future_ball_location.x, future_ball_location.y, car_location.z)
            controller = self.go_to_target(car_location, car_velocity, aerial_position)
            
            # If we're under the ball and ball is at right height, jump
            if (aerial_position - car_location).flat().length() < 50 and future_ball_location.z > 300:
                controller.jump = True
                controller.boost = True
                controller.pitch = -0.5  # Tilt up for aerial
                
                # If second jump still available and we're close, use it
                if car_location.z > 100 and future_ball_location.z - car_location.z < 300:
                    controller.jump = True
                    controller.pitch = -0.2
        else:
            # Can't or shouldn't aerial, go for normal shot
            return self.handle_shooting(car_location, car_velocity, ball_location, ball_velocity, enemy_goal)
            
        return controller

    def should_dribble(self, car_location, ball_location, ball_velocity):
        """Determine if we should dribble the ball"""
        # Check if ball is on ground and moving relatively slowly
        is_on_ground = ball_location.z < 120
        is_moving_slowly = ball_velocity.length() < 1000
        is_close = (ball_location - car_location).length() < 500
        
        return is_on_ground and is_moving_slowly and is_close

    def handle_dribble(self, car_location, car_velocity, ball_location, ball_velocity, enemy_goal):
        """Handle dribbling and flicking"""
        controller = SimpleControllerState()
        
        # Distance to ball
        distance = (ball_location - car_location).length()
        
        # Direction to enemy goal
        direction_to_goal = (enemy_goal - car_location).normalized()
        
        # Position slightly behind the ball in relation to the goal
        target_position = ball_location - direction_to_goal.scaled(self.DRIBBLE_DISTANCE)
        
        if (target_position - car_location).length() > 50:
            # Still getting in dribble position
            controller = self.go_to_target(car_location, car_velocity, target_position)
            controller.boost = False  # Don't boost while trying to control the ball
        else:
            # We're in dribble position
            self.dribbling = True
            
            # Match ball speed to maintain control
            controller.throttle = 1.0 if ball_velocity.length() > car_velocity.length() else 0.5
            
            # Steer toward goal
            controller.steer = steer_toward_target(self.get_car_facing_vector(self.get_latest_packet()), enemy_goal - car_location)
            
            # If close to enemy goal or defender approaching, flick
            if (enemy_goal - car_location).length() < 1500 or self.is_defender_approaching():
                # Flick sequence
                self.active_sequence = self.flick_sequence()
                return self.active_sequence.tick(self.get_latest_packet())
                
        return controller

    def flick_sequence(self):
        """Return a sequence for performing a flick"""
        sequence = Sequence([
            ControlStep(duration=0.1, controls=SimpleControllerState(throttle=1, jump=True)),
            ControlStep(duration=0.15, controls=SimpleControllerState(throttle=1)),
            ControlStep(duration=0.1, controls=SimpleControllerState(throttle=1, jump=True, pitch=-1)),
        ])
        return sequence

    def handle_shooting(self, car_location, car_velocity, ball_location, ball_velocity, target):
        """Handle normal ground shots"""
        controller = SimpleControllerState()
        
        # Predict ball position
        future_ball_location = ball_location + ball_velocity.scaled(0.2)
        
        # Calculate intercept point
        target_location = self.calculate_intercept_point(car_location, car_velocity, future_ball_location, ball_velocity, target)
        
        # Go to target location
        controller = self.go_to_target(car_location, car_velocity, target_location)
        
        # If close to ball, determine whether to shoot or set up
        if (ball_location - car_location).length() < 200:
            # Aim toward goal
            shot_vector = target - ball_location
            
            # If we have a good angle, shoot
            if self.has_good_shot_angle(car_location, ball_location, target):
                controller.boost = True
                
                # If fast enough, flip into the ball
                if car_velocity.length() > 1000:
                    self.active_sequence = self.shoot_flip_sequence(ball_location, target)
                    return self.active_sequence.tick(self.get_latest_packet())
        
        # Boost if far from the ball
        controller.boost = (ball_location - car_location).length() > 1000
        
        return controller

    def calculate_intercept_point(self, car_location, car_velocity, ball_location, ball_velocity, target):
        """Calculate where to intercept the ball for a good shot"""
        # Direction from ball to target
        ball_to_target = (target - ball_location).normalized()
        
        # How far behind the ball we want to hit it from
        offset_distance = 150
        
        # Aim for a point behind the ball relative to the target
        return ball_location - ball_to_target.scaled(offset_distance)

    def shoot_flip_sequence(self, ball_location, target):
        """Return a sequence for flipping into the ball toward the target"""
        # Calculate flip direction
        direction = (target - ball_location).normalized()
        pitch = -1 if direction.y > 0 else 1
        
        sequence = Sequence([
            ControlStep(duration=0.1, controls=SimpleControllerState(throttle=1, boost=True, jump=True)),
            ControlStep(duration=0.1, controls=SimpleControllerState(throttle=1, boost=True)),
            ControlStep(duration=0.3, controls=SimpleControllerState(throttle=1, jump=True, pitch=pitch)),
        ])
        return sequence

    def has_good_shot_angle(self, car_location, ball_location, target):
        """Determine if we have a good angle for a shot"""
        car_to_ball = (ball_location - car_location).normalized()
        ball_to_target = (target - ball_location).normalized()
        
        # Dot product of vectors - higher value means better alignment
        dot_product = car_to_ball.dot(ball_to_target)
        
        return dot_product > 0.7  # Roughly 45 degree angle or less

    def is_defender_approaching(self):
        """Check if an opponent is approaching to challenge the dribble"""
        packet = self.get_latest_packet()
        
        for i in range(packet.num_cars):
            if packet.game_cars[i].team != self.team:
                opponent_location = Vec3(packet.game_cars[i].physics.location)
                opponent_velocity = Vec3(packet.game_cars[i].physics.velocity)
                my_location = Vec3(packet.game_cars[self.index].physics.location)
                
                # Check if opponent is approaching
                distance = (opponent_location - my_location).length()
                approaching = opponent_velocity.dot((my_location - opponent_location).normalized()) > 500
                
                if distance < 1500 and approaching:
                    return True
                    
        return False

    def find_nearest_boost(self, car_location):
        """Find the nearest available boost pad"""
        closest_distance = float('inf')
        closest_boost = None
        
        for boost in self.boost_pad_tracker.boost_pads:
            if boost.is_active:
                distance = (Vec3(boost.location) - car_location).length()
                
                # Prefer full boosts but take small ones if closer
                if boost.is_full_boost:
                    distance *= 0.5
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_boost = Vec3(boost.location)
                    
        return closest_boost

    def is_critical_situation(self, car_location, ball_location, own_goal):
        """Check if the situation requires immediate attention"""
        ball_to_goal_distance = (ball_location - own_goal).length()
        car_to_ball_distance = (ball_location - car_location).length()
        
        # Ball moving toward our goal
        ball_packet = self.get_latest_packet().game_ball
        ball_velocity = Vec3(ball_packet.physics.velocity)
        moving_to_goal = ball_velocity.dot((own_goal - ball_location).normalized()) > 300
        
        return (ball_to_goal_distance < 2000 and moving_to_goal) or car_to_ball_distance < 500

    def predict_time_to_reach(self, car_location, car_velocity, target_location):
        """Predict time to reach target"""
        distance = (target_location - car_location).length()
        current_speed = car_velocity.length()
        
        # Simplified prediction
        if current_speed < 100:
            return distance / 1400  # Assuming max speed
        else:
            return distance / current_speed

    def go_to_target(self, car_location, car_velocity, target):
        """Go to a target position using appropriate throttle and steering"""
        controller = SimpleControllerState()
        
        # Direction to target
        car_to_target = target - car_location
        
        # Distance to target
        distance = car_to_target.length()
        
        # Car's forward direction
        car_orientation = self.get_car_facing_vector(self.get_latest_packet())
        
        # Calculate steering
        controller.steer = steer_toward_target(car_orientation, car_to_target)
        
        # Throttle and boost control
        controller.throttle = 1.0
        
        # Boost if we should go fast
        controller.boost = distance > 1500 and abs(controller.steer) < 0.3
        
        # Powerslide for sharp turns
        if abs(controller.steer) > 0.9 and car_velocity.length() > 500:
            controller.handbrake = True
        
        return controller

    def get_car_facing_vector(self, packet):
        """Get a vector pointing in the direction the car is facing"""
        car = packet.game_cars[self.index].physics.rotation
        pitch = float(car.pitch)
        yaw = float(car.yaw)
        
        facing_x = math.cos(pitch) * math.cos(yaw)
        facing_y = math.cos(pitch) * math.sin(yaw)
        
        return Vec3(facing_x, facing_y, 0)

    def get_latest_packet(self):
        """Get the latest game packet"""
        return self.get_game_tick_packet()
