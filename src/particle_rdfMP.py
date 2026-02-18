import numpy as np
from scipy.spatial import KDTree

import dask.array as da
from dask.distributed import Client
from dask.distributed import fire_and_forget

class RDFCalculator():
    def __init__(self):
        self.client = Client()
        
        
    def TreeRDF(self, positionArray, targetID, maxRange):
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

        indexArray = tree.query_ball_point(positionArray[targetID], maxRange)

        distanceArray = np.linalg.norm(positionArray[indexArray] -
                                       positionArray[targetID], axis=1)

        return distanceArray

    

    def FileTrawler(self, filename, frames, targetID, maxRange):
        
        self.broadenedPeaks = np.empty((len(frames), 5000))
        self.x = np.linspace(0, maxRange, 5000)
        
        with open(filename, 'r') as f:
            
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
            frameTotal = len(frames)
            frameCounter = 1
            continues = 0   
            
            for line in f:
            
                if collectData:

                    #The start of each frame contains header lines that must 
                    #be skipped
                    if continues != 0:
                        continues -= 1
                        continue

                    # ITEM: TIMESTEP is the first line after the end of data 
                    #for a frame
                    if line == "ITEM: TIMESTEP\n":
                        # yield the data for computation, wait for next 
                        #command, then erase old data
                        fire_and_forget(self.client.submit(self.PositionCruncher,
                                           positionArray, targetID,
                                           maxRange, frameCounter-1))
                        
                        if frameCounter > frameTotal:
                            break
                            
                        frameCounter+=1
                        positionArray = np.empty((atomCount, 3))
                        continues = 8

                    else:
                        lineList = line.split(" ")

                    for i in range(3):
                        positionArray[int(lineList[0])-1, i] = (
                            float(lineList[2+i])*simDimension)


                else:
                    if line == "ITEM: TIMESTEP\n":
                        if int(f.readline()) == frames[0]:

                            collectData = True
                            continues = 7

    def PositionCruncher(self, positionArray, targetID, maxRange, frameCount):
        distanceArray = self.TreeRDF(positionArray, targetID, maxRange)

        x = np.linspace(0, maxRange, 5000)
        y = np.zeros(5000)

        sigma = 0.05
        gaussianFunc = lambda x, mu : np.exp(-((x-mu)**2)/(2*sigma**2))
        for point in distanceArray:

            y += gaussianFunc(x, point)

        self.broadenedPeaks[frameCount] = y
        # print(f"{(i+1)/len(frames)*100:5.2f}" + "% complete", end='\r')