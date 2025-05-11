
# imports
import pandas as pd
import requests
import json
import time
import numpy as np
from matplotlib import pyplot as plt

class AnalyticBase:
    def change(self,a):
        return str(int(100000*(a[0]+a[1]/60+a[2]/3600))/100000)
    def typeoftime(self,timecode):
        if timecode==24:
            return "daily"
        elif timecode==0.25:
            return "minutely_15"
        else:
            return "hourly"
    def typeofFeature(self,feature):
        if feature:
            return 'forecast_days'
        else:
            return 'past_days'
    def combine(self,l1="0",l2="0",types=["temperature_2m"],timecode=1,feature=True,forcastdays=14):
            return 'https://api.open-meteo.com/v1/forecast?latitude='+l1+'&longitude='+l2+'&'+self.typeoftime(timecode)+"="+",".join(types)+"&"+self.typeofFeature(feature)+"="+str(forcastdays)
    def converttostring(self,listan):
        return ",".join(listan)
    def gethourlyparams1(self):
        return ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "pressure_msl", "surface_pressure", 
                "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", 
                "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m", 
                "shortwave_radiation", "direct_radiation", "direct_normal_irradiance", "diffuse_radiation", "global_tilted_irradiance"]

    def gethourlyparams2(self):
        return ["vapour_pressure_deficit", "cape", "evapotranspiration", "et0_fao_evapotranspiration", "precipitation", "snowfall", 
                "precipitation_probability", "rain", "showers", "weather_code", "snow_depth", "freezing_level_height", "visibility", 
                "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", 
                "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm", "is_day"]


    def getdailyparams(self):
        return ["temperature_2m_max","temperature_2m_min","apparent_temperature_max","apparent_temperature_min","precipitation_sum",
                "rain_sum","showers_sum","snowfall_sum","precipitation_hours","precipitation_probability_max",
                "precipitation_probability_min","precipitation_probability_mean","weather_code","sunrise","sunset","sunshine_duration",
                "daylight_duration","wind_speed_10m_max","wind_gusts_10m_max","wind_direction_10m_dominant","shortwave_radiation_sum",
                "et0_fao_evapotranspiration","uv_index_max","uv_index_clear_sky_max"]

    def getdata(self,l1="0",l2="0",types=["temperature_2m"],timecode=1,feature=True,forcastdays=14):
        return requests.get(self.combine(l1,l2,types,timecode,feature,forcastdays)).json()

    def dumpadatan(self,dat,filename="Data.json"):
        with open(filename,"w") as file:
            json.dump(dat,file)
    def readdatan(self,filename): 
        with open(filename,"r") as file:
            filen=file.read()
        return json.loads(filen)
    

class Analyticlowermid:
    def __init__(self):
        self.AB=AnalyticBase()
    def createnewfilewithparams(self,cordinater=[12,58],place="Nowere"):
        stringe=self.AB.gethourlyparams1()+self.AB.gethourlyparams2()
        print(place,cordinater)
        datan=self.AB.getdata(str(cordinater[0]),str(cordinater[1]),stringe,1,True,14)
        filenameis="./data/"+place+"_"+"_".join(["_".join(i.split(":")) for i in time.asctime().split(" ")])
        with open("./tempimpfile/latestfile.json","r") as file:
            listan=json.loads(file.read())     
        with open("./tempimpfile/latestfile.json","w") as file:
            if listan.get(place)==None:
                listan[place]=[filenameis+".json"]
            else:
                listan[place].append(filenameis+".json")
            json.dump(listan,file)  
        self.AB.dumpadatan(datan,filename=filenameis+".json")

    def handelfile(self,filename="./data/Bergstena15.46.json"):
        datan =self.AB.readdatan(filename)
        elevation=datan.get("elevation")
        data=datan.get("hourly")
        units=datan.get("hourly_units")
        return data,units,elevation
    
    def getlutning(time=0,day=0,lang=60,lati=0):#days since 1 jan 2024
        def f2(n):
            return np.arcsin(np.sin(-23.44/180*np.pi)*np.cos(2 * np.pi/365.24 *(n-0.473+10)+2 *0.0167*np.sin(2*np.pi/365.24 *(n-0.473-2))))
        result= np.pi/2-np.abs(np.pi/2-(np.pi/2+f2(day)-lang/180*np.pi)*np.sin(np.pi*2*(time/24+lati/360-0.25)))
        return (result+np.abs(result))/2
    def serchfile(self,name="",typen="cord"):
        match typen:
            case "altname":
                filename="./tempimpfile/cityfinderallcitysdatainv.json"
                mappen="./citysdatainv/"
            case "name":
                filename="./tempimpfile/cityfinderallcitysdata.json"
                mappen="./citysdata/"

            case _:
                filename="./tempimpfile/cityfinderallcitysdatacor.json"
                mappen="./citysdatacor/"
        with open(filename,"r") as file:
            lista=json.loads(file.read())
        lowerlimit=0
        upperlimit=len(lista)-1
        while upperlimit-lowerlimit>0:
            mid=(upperlimit+lowerlimit+1)//2
            if lista[mid]>name:
                upperlimit=mid-1
            else:
                lowerlimit=mid
        return mappen+"cityfinder_"+lista[upperlimit]+".json"            

    def addnewcity(self,name=""):
        with open("./tempimpfile/setofcities.json","r") as file:
            setofcities=json.loads(file.read())
        with open("./tempimpfile/alternativename.json","r") as file:
            alternativename=json.loads(file.read())
        with open(self.serchfile(name=name,typen="altname"),"r") as file:
            somealternativenamedatabase=json.loads(file.read())
        with open(self.serchfile(name=name,typen="cord"),"r") as file:
            somecoordinatesdatabase=json.loads(file.read())
        if setofcities.get(name)!=None:
            print(f"The city of _{name}_ does already exist")
            return False
        an=somealternativenamedatabase.get(name)[0]
        co=somecoordinatesdatabase.get(name)[0]
        if an!=None and co!=None:
            if co.get("Lat")!=None and co.get("Lon")!=None:
                for a in an:
                    alternativename[a]=name
                setofcities[name]=[co.get("Lat"),co.get("Lon")]
                with open("./tempimpfile/setofcities.json","w") as file:
                    json.dump(setofcities,file)
                with open("./tempimpfile/alternativename.json","w") as file:
                    json.dump(alternativename,file)
        else:
            print(f"The city of _{name}_ does not exist")
        return True
class Analyticuppermid:
    def __init__(self,filen="./data/Bergstena_Wed_Jan_29_23_50_59_2025.json"):
        self.AB=Analyticlowermid()
        self.filen=filen
        self.data,self.units,self.elevation=self.AB.handelfile(filen)
    
    def changetolatestfile(self):#inactive
        with open("./tempimpfile/latestfile.json","r") as file:
            lista=json.loads(file.read())
        if type(lista)==type([]) and len(lista)>1:
            filenn=lista[len(lista)-2]
            if filenn[len(filenn)-4:]==".json":
                self.filen=filenn
                self.data,self.units,self.elevation=self.AB.handelfile(self.filen)
            else:
                pass
    def changefile(self,filen):
        self.filen=filen
        self.data,self.units,self.elevation=self.AB.handelfile(self.filen)
    def ploter(self,wichone="temperature_2m",timefor=25):            
        if self.units.get(wichone)!=None:
            time1=self.data.get("time")
            temp=self.data.get(wichone)
        arrayen=np.array(temp[0:max(timefor,1)])
        plt.title(f"Diagram after {wichone} \n"+self.filen)
        plt.plot(arrayen,label="temperature")
        timerightnow=[int(i) for i in time.asctime().split()[3].split(":")]
        timerightnowint=(24+timerightnow[0]+time.timezone//3600)%24+timerightnow[1]/60+timerightnow[2]/3600
        lt=len(time1[0])
        #print(time1[0][lt-1-4:lt-1-2])
        plt.plot([timerightnowint,timerightnowint],[-0.1*np.max(arrayen)+1.1*np.min(arrayen),np.max(arrayen)],label=f"time now {(24+timerightnow[0]+time.timezone//3600)%24}:{timerightnow[1]}:{timerightnow[2]} [GMT]")
        plt.ylabel(f"{wichone} [{self.units.get(wichone)}]")
        plt.xlabel(f"tid från {time1[0]} [h]")
        plt.legend()
        plt.show()

class pictureisfun:
    def getallcontrycode(self,lan="en"):
        return requests.get("https://flagcdn.com/"+lan+"/codes.json").json()
    def getallcontrycodehalv(self,lan="en"):
        with open("./bilder/setlang.json","r") as file:
                ca_codes_in= json.loads(file.read())
        if ca_codes_in.get(lan)!=None:
            return requests.get("https://flagcdn.com/"+lan+"/codes.json").json()
    def getallcontrycodeadv(self,lan="en"):
        with open("./bilder/setlang.json","r") as file:
            lang= json.loads(file.read())
        #response_en=requests.get("https://flagcdn.com/en/codes.json").json()
        if lang.get(lan)!=None:
            with open("./bilder/ca_codes_in.json","r") as file:
                ca_codes_in= json.loads(file.read())
            if ca_codes_in.get(lan)==None:
                ca_codes_in[lan]=True
                info= requests.get("https://flagcdn.com/"+lan+"/codes.json").json()
                with open("./bilder/ca_code/code"+lan+".json", 'w') as file:
                    json.dump(info,file)
                with open("./bilder/ca_codes_in.json", 'w') as file:
                    json.dump(ca_codes_in,file)
    def getcontrypicture(self,contryname="se",size="w80",formats="png"):
        if formats not in set(["png","jpg"]):
            formats="png"
        if size not in set(["w20""w40", "w80","w160","w320","w640","w1280","w2560","h20","h24","h40","h60","h80","h120","h240"]):
            size="w20"
        url=("https://flagcdn.com/"+size+"/"+contryname.lower()+"."+formats)
        response=requests.get(url)
        #typ aigenererat nedan (men fint?)
        if response.status_code == 200:
            with open("./bilder/size"+size+"/somename_"+contryname+"_"+size+"."+formats, 'wb') as file:
                file.write(response.content)
            print(f"Image successfully downloaded: {'./bilder/size'+size+'/somename_'+contryname+'_'+size+'.'+formats}")
        else:
            print("Failed to retrieve image")
    def createallpictures(self,size="w160",formats="png"):
        allcode=self.getallcontrycode()
        for i in self.getallcontrycode():
            self.getcontrypicture(i,size=size,formats=formats)
        with open("./bilder/allcode.json", 'w') as file:
                json.dump(allcode,file)
    def formatsome(self):
        with open("./bilder/some.txt","r",encoding="utf-8") as file:
            lista=[i.split("\t") for i in file.read().split("\n")]
        emptydict={}
        emptysetdict={}
        for i in lista:
            emptydict[i[2]]={"english":i[0],"native":i[1]}
            emptysetdict[i[2]]=True
        with open("./bilder/dictlang.json","w") as file:
            json.dump(emptydict,file)
        with open("./bilder/setlang.json","w") as file:
            json.dump(emptysetdict,file)
def running():
    print("Will you continue?")
    with open("./tempimpfile/latestfile.json","r") as file:
        hashmap=json.loads(file.read())
    while input().lower()=="yes":
        
        while True:
            print("Vilken stad")
            inp= input()
            if inp=="":
                return 
            elif hashmap.get(inp)!=None:
                Ab.changefile(hashmap.get(inp)[len(hashmap.get(inp))-1])
                break
        while True:    
            print("which data?")
            datafors=input()
            if type(Ab.data)==type({"A":1}) or datafors=="":
                if Ab.data.get(datafors)!=None:
                    break
        if datafors=="":
            break
        print("which time?")
        timefors=input()
        try:
            s=int(timefors)
        except:
            pass
        else:
            Ab.ploter(datafors,s)
        print("Will you continue?")

def transformfirstcsvtotwolists(filename="./oldata/cities.csv",newname="./oldata/namecopy.csv",newcord="./oldata/corcopy.csv",sepold=";",newsep=";"):
    citys=pd.read_csv(filename,sep=sepold)
    print(citys)
    print(citys["Alternate Names"])
    emptylist1=[]
    emptylist3=[]
    for i in range(len(citys["Alternate Names"])):
        emptylist1.append([citys["Name"][i],citys["ASCII Name"][i]])

        emptylist3.append(citys["ASCII Name"][i])
        if type(citys["Alternate Names"][i])!=type(.1):
            for j in citys["Alternate Names"][i].split(", "):
                emptylist1.append([j,citys["ASCII Name"][i]])
    print(emptylist1)
    pd.DataFrame(emptylist1).to_csv(newname, sep=newsep,header=["Key","Value"], index=False,encoding='utf-8')
    df = citys.drop(citys.columns[[0, 2]], axis=1)
    pd.DataFrame(citys.drop(citys.columns[[0, 2]], axis=1)).to_csv(newcord, sep=newsep,header=["Name","Pop","Nation","Lat","Long"], index=False,encoding='utf-8')
    pd.DataFrame()
def askaboutnewplace():
    print("Will you continue with new place?")
    while input().lower()=="yes":
        print("Name:")
        namein=input()
        while True:
            if Ab.AB.addnewcity(namein):               
                print("success")
                break
            print("dosnät exist. Try again. Will you try")
            if input().lower()!="yes":
                break
def loadnewvalues():
    print("new values")
    if input().lower()=="yes":
        print("place") 
        inputname=input()
        with open("./tempimpfile/setofcities.json","r") as file:
            coordinates=json.loads(file.read())
        coordinate=coordinates.get(inputname)
        if coordinate!=None:
            print(coordinate)
            Ab.AB.createnewfilewithparams(coordinate,inputname)
        else:
            print("dosnät find them")

#pic=pictureisfun()
#pic.formatsome()
#pic.getallcontrycodeadv("en")
#pic.createallpictures(size="w160",formats="png")
#pic.createallpictures(size="h80",formats="png")
#pic.createallpictures(size="w640",formats="png")

#Ab.changetolatestfile()
#print("_"+"_".join(["_".join(i.split(":")) for i in time.asctime().split(" ")])+"_")
#print(Ab.AB.serchfile("Zzzz"))
#print("G\u00c3\u00b6teborg")
#Ab.AB.addnewcity("Goeteborg")
#Ab.getsome()

 

#its make some funny stuff if you say yes on them
Ab=Analyticuppermid()
askaboutnewplace()
loadnewvalues()
running()

