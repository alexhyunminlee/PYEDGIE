from ochre.Simulator import Simulator


class Building(Simulator):  # type: ignore[no-any-unimported]
    def __init__(self, name: str, region:str, attached:bool, characteristics:dict) -> None:
        super().__init__()
        self.name = name
        self.region = region
        self.attached = attached
        self.characteristics = characteristics
        self.calculateR()
    # Getters
    def getRegion(self):
        return self.region
    def getAttached(self):
        return self.attached
    def getStoryHeight(self):
        return self.characteristics['storyHeight']
    def getAspectRation(self):
        return self.characteristics['aspectRatio']
    def getNumStories(self):
        return self.characteristics['numStories']
    def getFloorArea(self):
        return self.characteristics['floorArea']
    def getR(self):
        return self.characteristics['R']
    def getDesignTempCool(self):
        return self.characteristics['deisgnTempCool']
    def getDesignTempHeat(self):
        return self.characteristics['deisgnTempHeat']

    # Setters
    def setRegion(self, region:str):
        self.region = region
    def setAttached(self, attached:bool):
        self.attached = attached
    def setStoryHeight(self, storyHeight:float):
        self.characteristics['storyHeight'] = storyHeight
    def setAspectRation(self, aspectRatio:float):
        self.characteristics['aspectRatio'] = aspectRatio
    def setNumStories(self, numStories:int):
        self.characteristics['numStories'] = numStories
    def setFloorArea(self, floorArea:float):
        self.characteristics['floorArea'] = floorArea
    def setR(self, R:float):
        self.characteristics['R'] = R
    def setDesignTempCool(self, designTempCool:float):
        self.characteristics['designTempCool'] = designTempCool
    def setDesignTempCool(self, designTempHeat:float):
        self.characteristics['designTempCool'] = designTempHeat

    def generateBaselineElectricity(self, tStart:datetime, tEnd:datetime, tWindow:timedelta, filePath:str):
        """
        Read the baseline electricity load file, scale it based on the building characteristics,
        and choose the data to be between the desired start and end time.

        Authors: Priyadarshan (priyada@purdue.edu), Alex Lee (alexlee5124@gmail.com), Zachary Tan
        Date: 2/4/2025

        """
        
        # Error checking:
        # 1) self.region should have appropriate values and self.attached should not be null.
        if self.region != ("West" or "Midwest" or "South" or "Northeast"):
            raise ValueError("generateBaselineElectricity: Please have an appropriate region for your building") 
        if self.attached is None:
            raise ValueError("generateBaselineElectricity: Please set the attached/detached status for your building")
        # 2) tStart, tEnd, tWindow should not be null and tStart < tEnd
        if (tStart is None) or (tEnd is None) or (tWindow is None):
            raise ValueError("generateBaselineElectricity: Please provide appropriate time parameters")
        if not (tStart < tEnd):
            raise ValueError("generateBaselineElectricity: The start time should be before the end time")
        
        # Read in raw data
        try:
            scaledData = pd.read_excel(filePath)
        except FileNotFoundError:
            print(f"Error: The file '{filePath}' was not found. Please check the file path.")
        except ValueError as e:
            print(f"Error: There was an issue with the file format. Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        # Randomly select a column and update data to only include that column
        scaledData = scaledData.iloc[:,[0,randrange(1,scaledData.shape[1])]]

        #set scaling factor by region
        if (self.region == "West"): #https://www.eia.gov/consumption/residential/data/2020/c&e/pdf/ce4.2.pdf
            scaling_detached = (17e9+96e9)/(16.97e6*8760)
            scaling_attached = ((1e9+6e9)/(8760*1.69e6) + (1e9+5e9)/(8760*1.89e6) + (3e9+14e9)/(8760*5.70e6) )/3
        elif (self.region == "MidWest"):
            scaling_detached = (18e9+115e9)/(8760*18.58e6)
            scaling_attached = ((1e9+6e9)/(8760*1.33e6) + (1e9+5e9)/(8760*1.95e6) + (2e9+10e9)/(8760*4.20e6) )/3
        elif (self.region == "NorthEast"):
            scaling_detached = (10e9+66e9)/(11.23e6*8760)
            scaling_attached = ((2e9+8e9)/(8760*1.95e6) + (2e9+8e9)/(8760*3.15e6) + (3e9+11e9)/(8760*5.10e6) )/3
        else:
            scaling_detached = (31e9+205e9)/(30.29e6*8760)
            scaling_attached = ((2e9+11e9)/(8760*2.48e6) + (1e9+8e9)/(8760*2.36e6) + (5e9+26e9)/(8760*7.83e6) )/3

        #scale data
        scaledData.update(scaledData.iloc[:, 1:].div(scaledData.iloc[:, 1:].mean(axis=0),axis="columns"))
        
        if (self.attached == True):
            scaledData.update(scaledData.iloc[:, 1:].mul(scaling_attached))
        else:
            scaledData.update(scaledData.iloc[:, 1:].mul(scaling_detached))

        # Convert UTC to EST time
        scaledData.insert(loc = 1,
          column = 'DateTime',
          value = scaledData['DateTimeUTC'] - timedelta(hours=5))
        scaledData.drop(columns=['DateTimeUTC'], inplace=True)
        
        baseline = pd.DataFrame(columns=scaledData.columns)
        
        # Create time series from tStart to tEnd
        scaledData['DateTime'] += pd.DateOffset(years=(tStart.year - scaledData['DateTime'][0].year))
        for year in range(0, tEnd.year - tStart.year + 1):
            # Add the number of years to the DateTime column for each iteration
            scaledData['DateTime'] = scaledData['DateTime'] + pd.DateOffset(years=year)
            baseline = pd.concat([baseline, scaledData], ignore_index=True)

        baseline = baseline[(baseline["DateTime"] >= tStart.replace(hour=0,minute=0,second=0,microsecond=0)) & (baseline["DateTime"] <= tEnd.replace(hour=0,minute=0,second=0,microsecond=0))]
        baseline.set_index(baseline.columns[0], inplace=True)
        
        # Retime 
        currentWindowSize = baseline.index.to_series().diff()[-1]
        if tWindow >= currentWindowSize:
            baseline = baseline.resample(tWindow).mean()
        else:
            baseline = baseline.resample(tWindow).interpolate(method='linear')
    
        return baseline


if __name__ == "__main__":
    building = Building("test_building")
