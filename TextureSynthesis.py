import numpy as np
#Image quilting Efros Freeman


def FindPatch(source, overlap, sampleSize, patchSize, overlapSize, isOverlapVertical, isOverlapHorizontal):
    patch = np.zeros((patchSize[0], patchSize[1],3))
    minDist = np.inf
    for _ in range(sampleSize):
        samplePatch = FindRandomPatch(source, patchSize)
        dist = GetDistance(overlap, samplePatch, overlapSize, isOverlapVertical, isOverlapHorizontal)
        if(dist<minDist) : 
            minDist = dist
            patch = samplePatch
    return patch

def FindRandomPatch(source, patchSize):
    y = np.random.randint(0, source.shape[0]-patchSize[0])
    x = np.random.randint(0, source.shape[1]-patchSize[1])
    return source[y:y+patchSize[0], x:x+patchSize[1], :]

def GetDistance(overlap, patch, overlapSize, isOverlapVertical, isOverlapHorizontal):
    patchMask = patch.copy()
    if isOverlapHorizontal and isOverlapVertical :  
        patchMask[overlapSize[0]:, overlapSize[1]:, :] = 0
    elif isOverlapVertical: 
        patchMask[:, overlapSize[1]:, :] = 0
    elif isOverlapHorizontal: 
        patchMask[overlapSize[0]:, :, :] = 0
    return np.sum((overlap - patchMask)**2)

def GetWindow(sourceWindowed, patch, overlapSize, isOverlapVertical, isOverlapHorizontal):
    patchMask = patch.copy()
    if isOverlapHorizontal and isOverlapVertical :  
        patchMask[overlapSize[0]:, overlapSize[1]:, :] = 0
    elif isOverlapVertical: 
        patchMask[:, overlapSize[1]:, :] = 0
    elif isOverlapHorizontal: 
        patchMask[overlapSize[0]:, :, :] = 0
    return np.sum((sourceWindowed - patchMask)**2, axis=2)

def ResolveHorizontal(error, overlapLength):
    #Horizontal cut
    horizontalPaths = []
    for k in range(overlapLength):
        horizontalOverlap = error[0:overlapLength, 0:error.shape[1]]

        horizontalPath = []
        horizontalPath.append(k)
        currentPosition = (k, 0)
        pathError = horizontalOverlap[currentPosition[0], currentPosition[1]];

        for i in range(1,error.shape[1]):
            j = currentPosition[0]
            column = [(j, i)]
            if(j>0): column.append((j-1, i))
            if(j<overlapLength-1): column.append((j+1, i))

            minError = float("inf")
            for k in range(len(column)):
                if(horizontalOverlap[column[k][0], column[k][1]]<minError) : 
                    minError = horizontalOverlap[column[k][0], column[k][1]]
                    currentPosition = column[k]
            pathError+= minError
            horizontalPath.append(currentPosition[0])
        horizontalPaths.append((pathError, horizontalPath))

    bestHorizontalPath = []
    minPathError = float("inf")
    for k in range(len(horizontalPaths)):
        if horizontalPaths[k][0] <minPathError:
            minPathError = horizontalPaths[k][0]
            bestHorizontalPath = horizontalPaths[k][1]

    return bestHorizontalPath 

def ResolveVertical(error, overlapLength):
    #Vertical cut
    verticalPaths = []
    for k in range(overlapLength):   
        verticalOverlap = error[0:error.shape[0], 0:overlapLength]

        verticalPath = []
        verticalPath.append(k)
        currentPosition = (0, k)
        pathError = verticalOverlap[currentPosition[0], currentPosition[1]];

        for i in range(1,error.shape[0]) :
            j = currentPosition[1]
            row = [(i, j)]
            if(j>0): row.append((i, j-1))
            if(j<overlapLength-1): row.append((i, j+1))

            minError = float("inf")
            for k in range(len(row)) :
                if(verticalOverlap[row[k][0], row[k][1]]<minError) : 
                    minError = verticalOverlap[row[k][0], row[k][1]]
                    currentPosition = row[k]
            pathError += minError
            verticalPath.append(currentPosition[1])
        verticalPaths.append((pathError, verticalPath))

    bestVerticalPath = []
    minPathError = float("inf")
    for k in range(len(verticalPaths)):
        if verticalPaths[k][0] <minPathError:
            minPathError = verticalPaths[k][0]
            bestVerticalPath = verticalPaths[k][1]

    return bestVerticalPath

def VerticalCut(error, overlapLength):
    
    verticalOverlap = error[0:error.shape[0], 0:overlapLength]

    overlapCuts = np.empty(verticalOverlap.shape, dtype=object)
    overlapCuts.fill((0, 0))
    for j in range(overlapLength):
        #On stock l'index du poids précedant pour faciliter la reconstruction du chemin (on evite de refaire des min comme ca )
        #Paire (valeur de chemins le plus court, index precedant dans le chemin = -1)
        overlapCuts[0, j] = (verticalOverlap[0, j], -1)

    for i in range(1, verticalOverlap.shape[0]):
        for j in range(0, overlapLength):
            # paires (valeur de chemins le plus court, index)
            row = [(overlapCuts[i-1, j][0], j)]
            if(j>0): row.append((overlapCuts[i-1, j-1][0], j-1))
            if(j<overlapLength-1): row.append((overlapCuts[i-1, j+1][0], j+1))
            minStep = min(row, key=lambda value: value[0])
            overlapCuts[i, j] = (minStep[0] + verticalOverlap[i, j], minStep[1])

    start = (float("inf"), 0)
    for i in range(overlapLength):
        if overlapCuts[-1, i][0] < start[0]:
            start = (overlapCuts[-1, i][0],i)

    currentLineCutIndex = start[1]
    verticalCut = []
    #parcours à l'envers de mes poids calculés
    for i in range(error.shape[0]-1, -1, -1):
        verticalCut.append(currentLineCutIndex)
        currentLineCutIndex = overlapCuts[i, currentLineCutIndex][1]#index suivant
    verticalCut.reverse();

    return verticalCut

def HorizontalCut(error, overlapLength):
    
    horizontalOverlap = error[0:overlapLength, 0:error.shape[1]]

    overlapCuts = np.empty(horizontalOverlap.shape, dtype=object)
    overlapCuts.fill((0, 0))
    for j in range(overlapLength):
        #On stock l'index du poids précedant pour faciliter la reconstruction du chemin (on evite de refaire des min comme ca )
        #Paire (valeur de chemins le plus court, index precedant dans le chemin = -1)
        overlapCuts[j, 0] = (horizontalOverlap[j, 0], -1)

    for i in range(1, horizontalOverlap.shape[1]):
        for j in range(0, overlapLength):
            # paires (valeur de chemins le plus court, index)
            row = [(overlapCuts[j, i-1][0], j)]
            if(j>0): row.append((overlapCuts[j-1, i-1][0], j-1))
            if(j<overlapLength-1): row.append((overlapCuts[j+1, i-1][0], j+1))
            minStep = min(row, key=lambda value: value[0])
            overlapCuts[j, i] = (minStep[0] + horizontalOverlap[j, i], minStep[1])


    start = (float("inf"), 0)
    for i in range(overlapLength):
        if overlapCuts[i, -1][0] < start[0]:
            start = (overlapCuts[i, -1][0],i)

    currentColumnCutIndex = start[1]
    horizontalCut = []
    #parcours à l'envers de mes poids calculés
    for i in range(error.shape[1]-1, -1, -1):
        horizontalCut.append(currentColumnCutIndex)
        currentColumnCutIndex = overlapCuts[currentColumnCutIndex, i][1]#index suivant
    horizontalCut.reverse();

    return horizontalCut


def ResolveOverlaps(sourceWindowed, patch, overlapSize, isOverlapVertical, isOverlapHorizontal):
    error = GetWindow(sourceWindowed, patch, overlapSize, isOverlapVertical, isOverlapHorizontal)
    if(isOverlapHorizontal):
        horizontalPath = HorizontalCut(error, overlapSize[0])
    if(isOverlapVertical):
        verticalPath = VerticalCut(error, overlapSize[1])
    #merge
    mergedPatch = np.zeros_like(patch)
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            verticalLimit =  verticalPath[i] if isOverlapVertical else -1
            horizontalLimit = horizontalPath[j] if isOverlapHorizontal else -1
            #TODO dans l'article initial ya pas de or ici c'est une diagonale
            if j<=verticalLimit or i<=horizontalLimit : 
                mergedPatch[i, j] = sourceWindowed[i,j]
            else:
                mergedPatch[i, j] = patch[i,j]
    #TODO si nécessaire lisser les pixels sur le chemin 
    return mergedPatch


def Quilt(source, imgSize, patchSize, overlapSize, patchSampleSize):
    source = source.astype(np.float64)
    generatedTexture = np.zeros((imgSize[0], imgSize[1], 3), dtype=source.dtype)
    initPatch = FindRandomPatch(source, patchSize)

    generatedTexture[0:patchSize[0], 0:patchSize[1]] = initPatch
    i = 0
    j = 1
    y = i*(patchSize[0]-overlapSize[0])
    x = j*(patchSize[1]-overlapSize[1])
    while y < imgSize[0] and imgSize[0]-y >overlapSize[0]:
        while x < imgSize[1] and imgSize[1]-x > overlapSize[1]:
            currentPatchSize = (min(patchSize[0], imgSize[0]-y), min(patchSize[1], imgSize[1]-x))
            overlap = generatedTexture[y:y+currentPatchSize[0], x:x+currentPatchSize[1]]
            patch = FindPatch(source, overlap, patchSampleSize, currentPatchSize, overlapSize, j>0, i>0)
            finalPatch = ResolveOverlaps(overlap, patch, overlapSize, j>0, i>0)
            generatedTexture[y:y+currentPatchSize[0], x:x+currentPatchSize[1]] = finalPatch
            j+=1
            x = j*(patchSize[1]-overlapSize[1])
        i+=1
        y = i*(patchSize[0]-overlapSize[0])
        j=0
        x=0
    return np.clip(generatedTexture,0,255).astype(np.uint8);

