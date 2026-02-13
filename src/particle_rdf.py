import numpy as np

def FrameImporter(filename, frame):
    """
    Extracts the position data for the given frame range.
    
    Parameters
    ----------
    filename : string
        The path to the position file
    frame : int or sequence of ints
        The target frame or sequence of frames to be extracted. If a sequence is used, all frames must be contiguous
        
    Yields
    ----------
    positionArray : nd.array
        A numpy array containing the positions of each particle in a frame. The ID of the particle is the index of the array.
    """
    
    with open(filename) as f:        
        
        # Get the number of atoms in the simulation as well as simulation dimension (assume cube)
        
        atomCount = 0
        simDimension = 0
        
        for line in f:
            if line == "ITEM: NUMBER OF ATOMS\n":
                atomCount = int(f.readline())
            if line == "ITEM: BOX BOUNDS pp pp pp\n":
                simDimension = float(f.readline().split(" ")[1])
                break
        
        # Move through the file and collect the position data for the given frames
        
        f.seek(0)
        positionArray = np.empty((atomCount, 3))
        
        collectData = False
        continues = 0         
        
        for line in f:
            
            if collectData:
                
                #The start of each frame contains header lines that must be skipped
                if continues != 0:
                    continues -= 1
                    continue
                
                # ITEM: TIMESTEP is the first line after the end of data for a frame
                if line == "ITEM: TIMESTEP\n":
                    # yield the data for computation, wait for next command, then erase old data
                    yield positionArray
                    positionArray = np.empty((atomCount, 3))
                    continues = 8
                
                else:
                    lineList = line.split(" ")
                
                for i in range(3):
                    positionArray[int(lineList[0])-1, i] = float(lineList[2+i])*simDimension
                
            
            else:
                if line == "ITEM: TIMESTEP\n":
                    if int(f.readline()) == frame[0]:
                    
                        collectData = True
                        continues = 7

def ParticleRDF(positionArray, targetID, binCount=1000, maxRange=20):
    """
    Calculates the histogram of distances between the target particle and every other particle in a frame.
    
    Parameters
    ----------
    positionArray : nd.array
        numpy array containing the position data of each particle. The ID should be the index.
    targetID : int
        The ID of the particle around which the rdf will be calculated.
    binCount : int, default=1000
        The number of bins in the rdf, defaults to 1000.
    maxRange : float, default=20
        The distance in angstroms the rdf should be calculated to.
    
    Returns
    ----------
    histValues : nd.array
        The values of each bins, normalized to a sum of 1.
    binEdges : nd.array of dtype float
        The edges of the histogram bins.
    """
    
    vector1 = positionArray[targetID]
    
    distanceArray = np.empty(np.shape(positionArray)[0])
    
    for c, vector2 in enumerate(positionArray):
        distance = np.linalg.norm(vector1-vector2)
        
        distanceArray[c] = distance
    
    histValues, binEdges = np.histogram(distanceArray, bins=binCount, density=True, range=(0, maxRange))
    
    return (histValues, binEdges)

def GeneratorWrapper(filename, frames, targetID):
    """
    Wrapper to minimize memory usage while generating the rdf for a particle over a range of frames. Does not save the position data after calculating the rdf
    
    Parameters
    ----------
    filename : string
        The path to the position file
    frame : int or sequence of ints
        The target frame or sequence of frames to be extracted. If a sequence is used, all frames must be contiguous
    targetID : int
        The ID of the particle around which the rdf will be calculated.    
    Returns
    ----------
    histValuesList : List of nd.arrays
        A list containing the values of each bins, normalized to a sum of 1, for each frame
    binEdgesList : List of nd.arrays of dtype float
        A list containing the edges of the histogram bins for each frame
    """
    frameGenerator = FrameImporter(filename, frames)
    
    histValuesList = []
    binEdgesList = []

    for i in range(len(frames)):
        positions = next(frameGenerator)

        histValues, binEdges = ParticleRDF(positions, targetID, binCount=500)
        histValuesList.append(histValues)
        binEdgesList.append(binEdges)

        print(f"{(i+1)/len(frames)*100:5.2f}" + "% complete", end='\r')
    
    return (histValuesList, binEdgesList)

def GeneratorWrapperGaussian(filename, frames, targetID):
    """
    Wrapper to minimize memory usage while generating the rdf for a particle over a range of frames.
    Does not save the position data after calculating the rdf. Additionally, points are broadened using gaussians and summed together.
    
    Parameters
    ----------
    filename : string
        The path to the position file
    frame : int or sequence of ints
        The target frame or sequence of frames to be extracted. If a sequence is used, all frames must be contiguous
    targetID : int
        The ID of the particle around which the rdf will be calculated.    
    Returns
    ----------
    x : nd.array
        Array covering the x axis containing 5000 points.
    yList : List of nd.arrays of dtype float
        A list containing the gaussian broadened signal at each distance.
    """
    
    frameGenerator = FrameImporter(filename, frames)
    
    yList = []

    for i in range(len(frames)):
        positions = next(frameGenerator)
        distanceArray = np.empty(np.shape(positions)[0])
        vector1 = positions[targetID]
        for c, vector2 in enumerate(positions):
            dist = np.linalg.norm(vector1-vector2)
            distanceArray[c] = dist
        
        x = np.linspace(0, 20, 5000)
        y = np.zeros(5000)
        
        sigma = 0.05
        gaussianFunc = lambda x, mu : np.exp(-((x-mu)**2)/(2*sigma**2))
        for point in distanceArray:
            
            y += gaussianFunc(x, point)
            
        yList.append(y)
        print(f"{(i+1)/len(frames)*100:5.2f}" + "% complete", end='\r')
        
    
    return (x, yList)