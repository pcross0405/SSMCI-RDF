import numpy as np
from scipy.spatial import KDTree

from dask.distributed import Client

class RDFCalculator():
    """
    Calculates the RDF for a single particle, based on targetID, over a range of
    molecular dynamics timesteps. The nearest neighbor peaks are gaussian
    broadened and the calculations use Dask for multiprocessing.

    Attributes
    ----------
    filename : string
        The path to the file to be processed.
    frames : list
        The frames that will be processed.
    targetID : int
        Which particle the rdf will be calculated for.
    maxRange : int
        The maximum distance of the rdf, in Angstroms.
    """
    def __init__(self, filename, frames, targetID, maxRange):
        self.client = Client()
        self.filename = filename
        self.frames = frames
        self.targetID = targetID
        self.maxRange = maxRange    

    def FileScanner(self):
        """
        Moves through an MD positions csv file, extracting the positions at
        each frame and calculating the rdf for the particle with the targetID.
        Uses Dask to multiprocess the calculations while the file continues
        reading.

        Returns
        ----------
        nd.array of floats
            Array containing the gaussian broadened RDFs for the targetID 
            particle over the supplied frame range.
        """
        self.x = np.linspace(0, self.maxRange, self.maxRange*250)
        
        futureList = [] # Stores futures that can be gathered at the end
        
        with open(self.filename, 'r') as f:
            
            # Acquire basic simulation details
            atomCount = 0
            simDimension = 0

            for line in f:
                if line == "ITEM: NUMBER OF ATOMS\n":
                    atomCount = int(f.readline())
                if line == "ITEM: BOX BOUNDS pp pp pp\n":
                    simDimension = float(f.readline().split(" ")[1])
                    break

            # Move through the file and collect the position data for 
            # each frame

            f.seek(0)
            positionArray = np.empty((atomCount, 3))

            collectData = False
            frameTotal = len(self.frames)
            frameCounter = 1
            continues = 0   
            
            for line in f:
            
                if collectData:

                    # The start of each frame contains header lines that must 
                    # be skipped
                    if continues != 0:
                        continues -= 1
                        continue

                    # ITEM: TIMESTEP is the first line after the end of data 
                    # for a frame
                    if line == "ITEM: TIMESTEP\n":
                        # Send the position data for this timestep to a 
                        # processor, letting this processor continue moving 
                        # through the file.

                        futureList.append(self.client.submit(
                            RDFGenerator, positionArray, self.targetID, 
                            self.maxRange, frameCounter-1, self.x))
                        
                        print(f"{(frameCounter-1)/len(self.frames)*100:5.2f}" + 
                              "% complete", end='\r')
                        
                        # Stop moving through the file on the last frame
                        if frameCounter > frameTotal:
                            
                            return np.array(self.client.gather(futureList))
                            
                        frameCounter+=1
                        continues = 8
                        

                    else:
                        lineList = line.split(" ")

                        for i in range(3):
                            positionArray[int(lineList[0])-1, i] = (
                                float(lineList[2+i])*simDimension)


                # Searches for the first frame in the frame list, skipping lines
                # until it is found. Then data is allowed to be collected.
                else:
                    if line == "ITEM: TIMESTEP\n":
                    
                        if int(f.readline()) == self.frames[0]:
                            # print("Reached Data")
                            collectData = True
                            continues = 7
                

def TreeRDF(positionArray, targetID, maxRange):
        """
        Calculates the histogram of distances between the target particle 
        and every other particle in a frame.

        Parameters
        ----------
        positionArray : nd.array
            numpy array containing the position data of each particle. The ID 
            should be the index.
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

        indexArray = tree.query_ball_point(positionArray[targetID],
                                           maxRange)

        distanceArray = np.linalg.norm(positionArray[indexArray] -
                                       positionArray[targetID], axis=1)

        return distanceArray
                
                
                
def RDFGenerator(positionArray, targetID, maxRange, frameCount, x):
    """
        Calls TreeRDF to generate the rdf and then broadens each peak using a
        guassian function. The sum of these gaussians is then returned.

        Parameters
        ----------
        positionArray : nd.array
            numpy array containing the position data of each particle. The ID 
            should be the index.
        targetID : int
            The ID of the particle around which the rdf will be calculated.
        maxRange : float
            The distance in angstroms the rdf should be calculated to.
        frameCount : int
            The index of the current frame. Starts at 0 and increments by 1.
        x : nd.array
            numpy array containing the x values used for the broadening.

        Returns
        ----------
        distanceArray : nd.array of floats
            Array containing the distances of all particles within maxRange.
        """
    distanceArray = TreeRDF(positionArray, targetID, maxRange)

    y = np.zeros(np.shape(x))

    sigma = 0.05
    gaussianFunc = lambda x, mu : np.exp(-((x-mu)**2)/(2*sigma**2))
    
    for point in distanceArray:

        y += gaussianFunc(x, point)

    return y
