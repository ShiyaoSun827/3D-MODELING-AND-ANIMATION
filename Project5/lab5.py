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

def interpolate_frames(frame1, frame2, n_transition_frames):
    
    #Generate n_transition_frames transition frames using linear interpolation
    #between two motion capture frames.
    
    transition_frames = []
    for frame in range(1, n_transition_frames + 1):
        interpolated_frame = {}
        # Calculate the fraction of the transition for this particular frame.
        fraction = frame / (n_transition_frames + 1)
        for joint in frame1:
            #we need to make sure the joint does exist in second frame or not
            if joint in frame2:
                # Linear interpolation 
                last_degrees = frame1[joint]
                current_degrees = frame2[joint]
                degree = []
                for m, n in zip(last_degrees, current_degrees):
                    # Apply linear interpolation formula for each angle.
                    frac = (1 - fraction) * m + fraction * n
                    # Append the interpolated angle to the list.
                    degree.append(frac)
                interpolated_frame[joint] = degree
        transition_frames.append(interpolated_frame)
    return transition_frames

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
    #re-arrange the order of the three motions
    random.shuffle(input_filenames)
    #a list to store the frames
    temp = []
    
    for index, name in enumerate(input_filenames):
        frames = parse_amc(name)  
        #generate n_transition_frames transition frames using interpolation between 
        #every two MoCap files
        if index > 0:
            transition_frames = interpolate_frames(temp[-1], frames[0], n_transition_frames)
            for i in transition_frames:
                temp.append(i)
        temp.extend(frames)
        

    with open(out_filename, 'w') as f:
        f.write(":FULLY-SPECIFIED\n:DEGREES\n")
        for frame_index, frame in enumerate(temp, start=1):
            f.write(f"{frame_index}\n")
            for joint, degrees in frame.items():
                degrees_str = " ".join(map(str, degrees))
                f.write(f"{joint} {degrees_str}\n")

def main():
    
    input_filenames = ['87_01.amc', '87_03.amc', '87_05.amc']
    n_transition_frames = 600
    concatMoCap(input_filenames, n_transition_frames, 'new.amc')


if __name__ == "__main__":
    main()