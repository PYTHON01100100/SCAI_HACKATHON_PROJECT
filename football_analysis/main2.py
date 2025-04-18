import os
import cv2
import numpy as np

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    video_path =  r"C:\Users\aalmyman\Videos\New folder (2)\football_analysis\input_videos\08fd33_4.mp4"
    
    # ✅ Check if video exists before reading
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at: {video_path}")
        return
    
    # Read Video
    video_frames = read_video(video_path)
    print(f"✅ Loaded {len(video_frames)} frames from video.")

    if not video_frames:
        print("❌ Error: No frames loaded. Check if the video file is valid and readable.")
        return

    # Initialize Tracker
    tracker = Tracker(r"C:\Users\aalmyman\Videos\New folder (2)\football_analysis\models\best.pt")

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=r"C:\Users\aalmyman\Videos\New folder (2)\football_analysis\stubs\track_stubs.pkl"
    )

    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path= r"C:\Users\aalmyman\Videos\New folder (2)\football_analysis\stubs\camera_movement_stub.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)

    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    output_path = r"C:\Users\aalmyman\Videos\New folder (2)\football_analysis\output_videos"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_video(output_video_frames, output_path)
    print(f"✅ Output video saved to: {output_path}")

if __name__ == '__main__':
    main()
