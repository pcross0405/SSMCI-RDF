import numpy as np
from scipy.spatial import KDTree

def FrameImporter(filename, frame):
    """
    Generator that extracts the position data for the given frame range. Each 
    time the function is called, the next frame will be output.
    
    Parameters
    ----------
    filename : string
        The path to the position file
    frame : int or sequence of ints
        The target frame or sequence of frames to be extracted. If a sequence 
        is used, all frames must be contiguous
        
    Yields
    ----------
    positionArray : nd.array
        A numpy array containing the positions of each particle in a frame. The 
        ID of the particle is the index of the array.
    """
    
    with open(filename) as f:        
        
        # Get the number of atoms in the simulation as well as simulation 
        # dimension (assume cube)
        
        atomCount = 0
        simDimension = 0
        
        for line in f:
            if line == "ITEM: NUMBER OF ATOMS\n":
                atomCount = int(f.readline())
            if line == "ITEM: BOX BOUNDS pp pp pp\n":
                simDimension = float(f.readline().split(" ")[1])
                break
        
        # Move through the file and collect the position data for each frame
        
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


def TreeRDF(positionArray, targetID, maxRange):
    """
    Calculates the histogram of distances between the target particle and every 
    other particle in a frame.
    
    Parameters
    ----------
    positionArray : nd.array
        numpy array containing the position data of each particle. The ID should
        be the index.
    targetID : int
        The ID of the particle around which the rdf will be calculated.
    maxRange : float
        The distance in angstroms the rdf should be calculated to.
    
    Returns
    ----------
    distanceArray : nd.array of floats
        Array containing the distances of all particles within maxRange.
    """
    tree = KDTree(positionArray)
    
    indexArray = tree.query_ball_point(positionArray[targetID], maxRange)
    
    distanceArray = np.linalg.norm(positionArray[indexArray] - positionArray[targetID], axis=1)
    
    return distanceArray

def GeneratorWrapper(filename, frames, targetID, maxRange):
    """
    Wrapper to minimize memory usage while generating the rdf for a particle 
    over a range of frames. Does not save the position data after calculating
    the rdf
    
    Parameters
    ----------
    filename : string
        The path to the position file
    frame : int or sequence of ints
        The target frame or sequence of frames to be extracted. If a sequence
        is used, all frames must be contiguous
    targetID : int
        The ID of the particle around which the rdf will be calculated.
    maxRange : float
        The distance in angstroms the rdf should be calculated to.
        
    Returns
    ----------
    histValuesArray : nd.array
        A 2d array organized by frame x histogramValues
    binEdgesList : nd.array
        A 2d array organized by frame x binEdges
    """
    frameGenerator = FrameImporter(filename, frames)
    
    histValuesArray = np.empty((len(frames), binCount))
    binEdgesArray = np.empty((len(frames), binCount))

    for i in range(len(frames)):
        positionArray = next(frameGenerator)

        distanceArray = TreeRDF(positionArray, targetID, maxRange)
        histValues, binEdges = np.histogram(distanceArray, density=True,
                                            binCount=1000, range=(0, maxRange))
        histValuesArray[i] = histValues
        binEdgesArray[i] = binEdges

        print(f"{(i+1)/len(frames)*100:5.2f}" + "% complete", end='\r')
    
    return (histValuesArray, binEdgesArray)


def GeneratorWrapperGaussian(filename, frames, targetID, maxRange):
    """
    Wrapper to minimize memory usage while generating the rdf for a particle 
    over a range of frames. Does not save the position data after calculating 
    the rdf. Additionally, points are broadened using gaussians 
    and summed together.
    
    Parameters
    ----------
    filename : string
        The path to the position file
    frame : int or sequence of ints
        The target frame or sequence of frames to be extracted. If a sequence 
        is used, all frames must be contiguous
    targetID : int
        The ID of the particle around which the rdf will be calculated.
    maxRange : float
        The distance in angstroms the rdf should be calculated to.
        
    Returns
    ----------
    x : nd.array
        Array covering the x axis containing 5000 points.
    broadenedPeaks : nd.array of dtype float
        A 2d array organized by frame x peakHeight
    """
    
    frameGenerator = FrameImporter(filename, frames)
    
    broadenedPeaks = np.empty((len(frames), 5000))

    for i in range(len(frames)):
        positionArray = next(frameGenerator)

        distanceArray = TreeRDF(positionArray, targetID, maxRange)
        
        x = np.linspace(0, maxRange, 5000)
        y = np.zeros(5000)
        
        sigma = 0.05
        gaussianFunc = lambda x, mu : np.exp(-((x-mu)**2)/(2*sigma**2))
        for point in distanceArray:
            
            y += gaussianFunc(x, point)
            
        broadenedPeaks[i] = y
        print(f"{(i+1)/len(frames)*100:5.2f}" + "% complete", end='\r')
        
    
    return (x, broadenedPeaks)