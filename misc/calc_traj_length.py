import numpy as np
from datetime import datetime

def read_trajectory_file(file_path, skip_first_line=True):
    """
    读取轨迹文件，每行格式为 x y z t (空格分隔)
    可以跳过第一行（例如标题行）
    返回轨迹点列表
    """
    trajectory = []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            # 如果需要跳过第一行
            if skip_first_line and lines:
                skipped_line = lines[0].strip()
                print(f"跳过的第一行: {skipped_line}")
                lines = lines[1:]
            
            for line_num, line in enumerate(lines, 2 if skip_first_line else 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 分割字段
                fields = line.split(' ')
                if len(fields) != 4:
                    print(f"警告: 第{line_num}行格式不正确: {line}")
                    continue
                
                try:
                    x = float(fields[0])
                    y = float(fields[1])
                    z = float(fields[2])
                    t = float(fields[3])  # 时间戳
                    
                    trajectory.append((x, y, z, t))
                    
                except ValueError as e:
                    print(f"警告: 第{line_num}行数据转换错误: {line}, 错误: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"错误: 文件未找到: {file_path}")
        return []
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
        return []
    
    print(f"成功读取 {len(trajectory)} 个轨迹点")
    return trajectory

def sort_trajectory_by_time(trajectory):
    """
    按时间戳对轨迹点进行排序
    """
    sorted_trajectory = sorted(trajectory, key=lambda point: point[3])  # 按时间戳(第4个元素)排序
    print(f"已按时间戳排序轨迹点")
    return sorted_trajectory

def segment_trajectory(trajectory, time_threshold=1.0):
    """
    将轨迹按照时间差分段
    当相邻点时间差大于time_threshold秒时，轨迹断开为不同段
    """
    if len(trajectory) < 2:
        return [trajectory] if trajectory else []
    
    segments = []
    current_segment = [trajectory[0]]
    
    for i in range(1, len(trajectory)):
        time_diff = trajectory[i][3] - trajectory[i-1][3]
        
        if time_diff <= time_threshold:
            # 时间差小于等于阈值，属于同一段
            current_segment.append(trajectory[i])
        else:
            # 时间差大于阈值，开始新的一段
            segments.append(current_segment)
            current_segment = [trajectory[i]]
    
    # 添加最后一段
    segments.append(current_segment)
    
    print(f"轨迹被分为 {len(segments)} 段")
    
    # 显示分段信息
    for i, segment in enumerate(segments):
        if len(segment) > 0:
            start_time = segment[0][3]
            end_time = segment[-1][3]
            print(f"  段 {i+1}: {len(segment)} 个点, 时间范围: {start_time:.2f} - {end_time:.2f}")
    
    return segments

def calculate_segment_lengths(segments):
    """
    计算每个轨迹段的长度
    """
    segment_lengths = []
    
    for i, segment in enumerate(segments):
        if len(segment) < 2:
            segment_lengths.append(0.0)
            continue
        
        segment_length = 0.0
        for j in range(1, len(segment)):
            x1, y1, z1, t1 = segment[j-1]
            x2, y2, z2, t2 = segment[j]
            
            # 计算两点间的欧几里得距离
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            segment_length += distance
        
        segment_lengths.append(segment_length)
        print(f"段 {i+1} 长度: {segment_length:.2f} 米")
    
    total_length = sum(segment_lengths)
    print(f"轨迹总长度: {total_length:.2f} 米")
    
    return segment_lengths, total_length

def analyze_trajectory(segments):
    """
    分析轨迹的基本信息
    """
    if not segments:
        print("轨迹数据为空")
        return
    
    # 合并所有段以计算总体统计信息
    all_points = []
    for segment in segments:
        all_points.extend(segment)
    
    if not all_points:
        print("没有有效的轨迹点")
        return
    
    # 提取坐标和时间戳
    x_coords = [point[0] for point in all_points]
    y_coords = [point[1] for point in all_points]
    z_coords = [point[2] for point in all_points]
    timestamps = [point[3] for point in all_points]
    
    print("\n=== 轨迹分析报告 ===")
    print(f"轨迹段数: {len(segments)}")
    print(f"总轨迹点数: {len(all_points)}")
    print(f"时间范围: {min(timestamps):.2f} - {max(timestamps):.2f}")
    print(f"X坐标范围: [{min(x_coords):.2f}, {max(x_coords):.2f}]")
    print(f"Y坐标范围: [{min(y_coords):.2f}, {max(y_coords):.2f}]")
    print(f"Z坐标范围: [{min(z_coords):.2f}, {max(z_coords):.2f}]")
    print(f"轨迹跨度: X:{max(x_coords)-min(x_coords):.2f}, Y:{max(y_coords)-min(y_coords):.2f}, Z:{max(z_coords)-min(z_coords):.2f}")
    
    # 分析每段轨迹
    print("\n=== 分段轨迹详情 ===")
    for i, segment in enumerate(segments):
        if len(segment) > 0:
            seg_x = [p[0] for p in segment]
            seg_y = [p[1] for p in segment]
            seg_z = [p[2] for p in segment]
            seg_t = [p[3] for p in segment]
            
            print(f"段 {i+1}:")
            print(f"  点数: {len(segment)}")
            print(f"  时间范围: {min(seg_t):.2f} - {max(seg_t):.2f}")
            print(f"  持续时间: {max(seg_t)-min(seg_t):.2f} 秒")
            print(f"  空间范围: X[{min(seg_x):.2f}, {max(seg_x):.2f}], Y[{min(seg_y):.2f}, {max(seg_y):.2f}], Z[{min(seg_z):.2f}, {max(seg_z):.2f}]")

def main():
    # 文件路径 - 请修改为实际文件路径
    file_path = "/home/ericxhzou/Data/IF1-Cloud.txt"  # 替换为你的轨迹文件路径
    
    print("开始处理轨迹文件...")
    
    # 1. 读取轨迹数据
    trajectory = read_trajectory_file(file_path)
    
    if not trajectory:
        print("无法读取轨迹数据，程序退出")
        return
    
    # 2. 按时间戳排序
    sorted_trajectory = sort_trajectory_by_time(trajectory)
    
    # 3. 将轨迹按时间差分段
    segments = segment_trajectory(sorted_trajectory, time_threshold=0.2)
    
    # 4. 计算各段长度和总长度
    segment_lengths, total_length = calculate_segment_lengths(segments)
    
    # # 5. 分析轨迹
    # analyze_trajectory(segments)
    
    # # 6. 打印每段的前几个点作为示例
    # print("\n=== 各段前几个轨迹点 ===")
    # for i, segment in enumerate(segments):
    #     if len(segment) > 0:
    #         print(f"段 {i+1} 的前{min(3, len(segment))}个点:")
    #         for j, point in enumerate(segment[:3]):
    #             print(f"  点{j+1}: x={point[0]:.2f}, y={point[1]:.2f}, z={point[2]:.2f}, t={point[3]:.2f}")
    #         if len(segment) > 3:
    #             print(f"  ... 还有 {len(segment)-3} 个点")


if __name__ == "__main__":
    main()
