import random

"""
Adapted from https://github.com/CalciferZh/AMCParser/blob/master/amc_parser.py
"""
def read_line(stream, idx):
    if idx >= len(stream):
        return None, idx
    line = stream[idx].strip().split()
    idx += 1
    return line, idx

def parse_amc(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        if line == ':DEGREES':
            content = content[idx+1:]
            break

    frames = []
    idx = 0
    line, idx = read_line(content, idx)
    assert line[0].isnumeric(), line
    EOF = False
    while not EOF:
        joint_degree = {}
        while True:
            line, idx = read_line(content, idx)
            if line is None:
                EOF = True
                break
            if line[0].isnumeric():
                break
            joint_degree[line[0]] = [float(deg) for deg in line[1:]]
        frames.append(joint_degree)
    return frames

def concatMoCap(input_filenames, n_transition_frames, out_filename):
    '''
    concatenate the input MoCap files in random order, 
    generate n_transition_frames transition frames using interpolation between every two MoCap files, 
    save the result as out_filename.
      parameter:
        input_filenames: [str], a list of all input filename strings
        n_transition_frames: int, number of transition frames
        out_filename: output file name
      return:
        None
    '''
    random.shuffle(input_filenames)  # Step 1: Randomize order
    
    all_frames = []  # To hold all frames including transitions
    
    for i, file_name in enumerate(input_filenames):
        frames = parse_amc(file_name)  # Step 2: Parse MoCap Files
        
        if i > 0:  # If not the first file, generate transition frames
            # Step 3: Interpolate Transition Frames
            last_frame_previous_file = all_frames[-1]
            first_frame_current_file = frames[0]
            for t in range(1, n_transition_frames + 1):
                interpolated_frame = {}
                fraction = t / (n_transition_frames + 1)
                for joint in last_frame_previous_file:
                    if joint in first_frame_current_file:
                        print(joint)
                        # Linear interpolation for each joint
                        last_degrees = last_frame_previous_file[joint]
                        current_degrees = first_frame_current_file[joint]
                        interpolated_degrees = [(1 - fraction) * ld + fraction * cd for ld, cd in zip(last_degrees, current_degrees)]
                        interpolated_frame[joint] = interpolated_degrees
                all_frames.append(interpolated_frame)
                
        all_frames.extend(frames)  # Add current file's frames to all_frames
        
    # Step 4 & 5: Write Output File
    with open(out_filename, 'w') as f:
        f.write(":FULLY-SPECIFIED\n:DEGREES\n")
        for frame_index, frame in enumerate(all_frames, start=1):
            f.write(f"{frame_index}\n")
            for joint, degrees in frame.items():
                degrees_str = " ".join(map(str, degrees))
                f.write(f"{joint} {degrees_str}\n")

def main():
    
    input_filenames = ['87_01.amc', '87_03.amc', '87_05.amc']
    # initialize 500 as n_transition_frames
    n_transition_frames = 500
    # initialize '18_concatenate.amc' as out_filename
    out_filename = 'combined.amc'
    # run concatMoCap
    concatMoCap(input_filenames, n_transition_frames, out_filename)


if __name__ == "__main__":
    main()