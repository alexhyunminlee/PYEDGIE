"""
Building.py : Class for buildings

This file contains the class definitions for a building. A building has a list of
properties describing its construction year, type, etc., a mechanical heating/cooling system,
a list of functions.

Authors: Priyadarshan (priyada@purdue.edu), Alex Lee (alexlee5124@gmail.com)
Date: 12/13/2024

"""


class Building:
    def __init__(self, region: str, attached: bool, characteristics: dict):
        """
        Initializes a Building instance.
        Parameters:
        - region: Building's census region, i.e. West, Midwest, Northeast, South.
        - attached: Attached/detached status of the building. True is attached, False is detached.
        - characteristics: Dictionary of house characteristics. Includes:
            - storyHeight: Height of each story in meters. Each floor is assumed to have the same height
            - aspectRatio: Aspect ratio of building's length/width
            - numStories: Number of stories for the building.
            - floorArea: Total floor area of the building in m^2

        """
        self.region = region
        self.attached = attached
        self.characteristics = characteristics

    # Getters
    def getRegion(self):
        return self.region

    def getAttached(self):
        return self.attached

    def getStoryHeight(self):
        return self.characteristics["storyHeight"]

    def getAspectRation(self):
        return self.characteristics["aspectRatio"]

    def getNumStories(self):
        return self.characteristics["numStories"]

    def getFloorArea(self):
        return self.characteristics["floorArea"]

    def getR(self):
        return self.characteristics["R"]

    def getDesignTempCool(self):
        return self.characteristics["deisgnTempCool"]

    def getDesignTempHeat(self):
        return self.characteristics["deisgnTempHeat"]

    # Setters
    def setRegion(self, region: str):
        self.region = region

    def setAttached(self, attached: bool):
        self.attached = attached

    def setStoryHeight(self, storyHeight: float):
        self.characteristics["storyHeight"] = storyHeight

    def setAspectRation(self, aspectRatio: float):
        self.characteristics["aspectRatio"] = aspectRatio

    def setNumStories(self, numStories: int):
        self.characteristics["numStories"] = numStories

    def setFloorArea(self, floorArea: float):
        self.characteristics["floorArea"] = floorArea

    def setR(self, R: float):
        self.characteristics["R"] = R

    def setDesignTempCool(self, designTempCool: float):
        self.characteristics["designTempCool"] = designTempCool

    def setDesignTempHeat(self, designTempHeat: float):
        self.characteristics["designTempHeat"] = designTempHeat

    def calculateR(self):
        """
        Calculate overall thermal resistance R parameter for the building.

        Authors: Priyadarshan (priyada@purdue.edu), Alex Lee (alexlee5124@gmail.com), Zachary Tan
        Date: 2/20/2025

        """

        storyHeight = self.characteristics["storyHeight"]
        aspectRatio = self.characteristics["aspectRatio"]
        numStories = self.characteristics["numStories"]
        floorArea = self.characteristics["floorArea"]

        lamb = 0.25
        Ur = 0.36
        density = 1.293  # kg/m^3
        Cp = 1005  # J/kg-K
        volume = floorArea * storyHeight  # m^3
        v = 1.5  # Air Exchange Rate per hour
        h_out = 22.7 / 1000
        h_in = 8.29 / 1000
        valueWindow = 2.5  # W/m^2-K
        valueWall = 0.4  # W/m^2-K

        if self.attached:
            # calculation for attached
            Aw = 2 * storyHeight * (aspectRatio + 1) * ((numStories * floorArea / aspectRatio) ** 0.5) / 2
        else:
            # Calculation for detached
            Aw = 2 * storyHeight * (aspectRatio + 1) * ((numStories * floorArea / aspectRatio) ** 0.5)
        # Modify this code
        areaRoof = floorArea / numStories
        modtCp = v * Cp * density * volume / 3600
        Uwall = lamb * valueWindow + (1 - lamb) * valueWall
        R = 1 / (modtCp / 1000 + Uwall * Aw / 1000 + Ur * areaRoof / 1000) + 1 / (h_in * Aw) + 1 / (h_out * Aw)
        self.characteristics["R"] = R
