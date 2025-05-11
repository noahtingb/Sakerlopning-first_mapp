import json
import pandas as pd

def transformfirstcsvtotwolists(filename="cities.csv",newname="namecopy1.json",newcord="corcopy1.csv",sepold=";",newsep=";",dumpadataname=False,dumpadatacord=False):
    citys=pd.read_csv(filename,sep=sepold)
    #print(citys)
    #print(citys["Alternate Names"])
    emptylist1={}
    emptylist3={}
    emptylist2={}

    listan1=[]
    listan3=[]
    #emptylist3=set([])
    names=["Name","Country Code","Population","Lat","Lon"]#["Name","ASCII Name","Alternate Names","Country Code","Population","Lat","Lon"]

    for i in range(len(citys["Alternate Names"])):
        if type(citys["Name"][i])==type(""):
            if emptylist1.get(citys["Name"][i])==None:
                emptylist1[citys["Name"][i]]=[citys["ASCII Name"][i]]
                listan1.append(citys["Name"][i])
            else:
                #print("caaaq5")
                emptylist1[citys["Name"][i]].append(citys["ASCII Name"][i])
            if emptylist1.get(citys["ASCII Name"][i])==None:
                emptylist1[citys["ASCII Name"][i]]=[citys["ASCII Name"][i]]           
                listan1.append(citys["ASCII Name"][i])
            else:
                #print("caaaq4")
                emptylist1[citys["ASCII Name"][i]].append(citys["ASCII Name"][i])        


            if type(citys["ASCII Name"][i])==type(""):
                nameen=citys["ASCII Name"][i]
                if emptylist3.get(nameen)==None:
                    listan3.append(nameen)
                if type(citys["Alternate Names"][i])==type(""):
                    if emptylist3.get(nameen)==None:
                        emptylist2[nameen]=[[citys["Name"][i]]+citys["Alternate Names"][i].split(", ")]
                    else:
                        #print("caaaq3")
                        emptylist2[nameen].append([citys["Name"][i]]+citys["Alternate Names"][i].split(", "))
                else:
                    if emptylist3.get(nameen)==None:
                        emptylist2[nameen]=[[citys["Name"][i]]]
                    else:
                        #print("caaaq2")
                        emptylist2[nameen].append([citys["Name"][i]])
                if emptylist3.get(nameen)==None:
                    emptylist3[nameen]=[{"Name":citys["Name"][i],"Country Code":citys["Country Code"][i],"Population":citys["Population"][i],"Lat":citys["Lat"][i],"Lon":citys["Lon"][i]}]
                else:
                    #print("caaaq")
                    emptylist3[nameen].append({"Name":citys["Name"][i],"Country Code":citys["Country Code"][i],"Population":citys["Population"][i],"Lat":citys["Lat"][i],"Lon":citys["Lon"][i]})

        #emptylist3.add(citys["ASCII Name"][i])

        if type(citys["Alternate Names"][i])!=type(.1):
            for j in citys["Alternate Names"][i].split(", "):
                if type(j)==type(""):
                    if emptylist1.get(j)==None:
                        emptylist1[j]=[citys["ASCII Name"][i]]
                        listan1.append(j)
                    else:
                        #print("caaa")
                        emptylist1[j].append(citys["ASCII Name"][i])

    if dumpadataname==False:
        index=0
        match index:
            case 0:
                listannn=listan1
                dictttt=emptylist1
                filepath="citysdata"
            case 1:
                listannn=listan3
                dictttt=emptylist2
                filepath="citysdatainv"

            case _:
                listannn=listan3
                dictttt=emptylist3
                filepath="citysdatacor"
        for i in range(len(listannn)):
            if type(listannn[i])!=type(""):
                print(i,listannn[i],type(i),dictttt[listannn[i]])
                listannn[i]="Nan"
                dictttt[listannn[i]]="Nan"
        
        intlistaforsplit=[]
      
        listannn.sort()
        l2=len(listannn)
        for i in range(min(100,l2//1000+1)):
            intlistaforsplit.append((i*l2)//min(100,l2//1000+1))
        intlistaforsplit.append(l2)
        listannn.append("else")
        somelistaforsplits=[listannn[i] for i in intlistaforsplit]
        for i in range(len(somelistaforsplits)-1):
            while len(somelistaforsplits[i].split("/"))>1 or  len(somelistaforsplits[i].split("\\"))>1:
                intlistaforsplit[i]+=1
                somelistaforsplits[i]=listannn[i]
                                                                                     
        for i in range(len(somelistaforsplits)-1):
            emptydict={}
            for j in listannn[intlistaforsplit[i]:intlistaforsplit[i+1]]:
                #if type(dictttt[j])==type("") or index!=0:
                    emptydict[j]=dictttt[j]
            print(filepath+f"\cityfinder_{somelistaforsplits[i]}.json")
            with open(filepath+f"\cityfinder_{somelistaforsplits[i]}.json","w") as fil:
                json.dump(emptydict,fil)
        with open(f"cityfinderall"+filepath+".json","w") as fil:
            json.dump(somelistaforsplits[:len(somelistaforsplits)-1],fil)

    
    #pd.DataFrame(emptylist1).to_csv(newname, sep=newsep,header=["Key","Value"], index=False,encoding='utf-8')
    #df = citys.drop(citys.columns[[0, 2]], axis=1)
    #pd.DataFrame(citys.drop(citys.columns[[0, 2]], axis=1)).to_csv(newcord, sep=newsep,header=["Name","Pop","Nation","Lat","Long"], index=False,encoding='utf-8')
    #pd.DataFrame()
    
    #with open(newname,"w") as fil:
    #    json.dump(emptylist1,fil)
#def matchsome(name):
#    if chr(name[0]):
#        pass
#print(ord("A"))
#print(ord("Z"))
print("\u00e4")
#transformfirstcsvtotwolists()