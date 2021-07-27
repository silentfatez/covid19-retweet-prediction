import category_encoders as ce
import pandas as pd


def hash_encode_username(df_t):
    encoder=ce.HashingEncoder(cols='Username',n_components=512,max_process=1)
    newdf=encoder.fit_transform(df_t)
    return newdf
def clean_timestamp(df_t):
    df_t['Timestamp']=pd.to_datetime(df_t['Timestamp'])
    return df_t
def create_entitylist(df_t):
    comblist=[]
    for i in df_t.Entities.str.split(';').tolist():
        newlist=[]
        for j in i:
            newstring=j.split(":")[-1]
            newlist.append(newstring)
            newlist = [0 if i=='null' or i=='' else i for i in newlist]
        newlist = newlist[:42]+ [0]*(42 - len(newlist[:42]))
        comblist.append(newlist[:-1])
    newdf=pd.DataFrame(comblist)
    df_t=df_t.drop('Entities',axis=1)
    df_t=pd.concat([df_t, newdf], axis=1)
    return df_t
            
def split_sentiment(df_t):
    newdf=pd.DataFrame(df_t.Sentiment.str.split('-').tolist())
    newdf = newdf.apply(pd.to_numeric)
    df_t=df_t.drop('Sentiment',axis=1)
    df_t=pd.concat([df_t, newdf], axis=1)
    return df_t
            
def hash_encode_column(x):
    encoder=ce.HashingEncoder(n_components=64,max_process=1)
    newdf=encoder.fit_transform(pd.DataFrame(x))
    return newdf

def clean_general(df_t,column):
    totallist=[]
    listtoprocess=df_t[column].str.split(' ').tolist()
    for j in listtoprocess:
        j = [None if i=='null;' or i=='' else i for i in j]
        j=j[:20]+[None]*(20-len(j[:20]))
        totallist.append(j)
    df=pd.DataFrame(totallist)
    df_t=pd.concat([df_t,hash_encode_column(df)], axis=1)
    return df_t
def count_general(df_t,column):
    df_t[column]=df_t[column].str.split(' ').str.len() 
    return df_t[(df_t[column]>270)==False]
    
def clean_url(df_t):
    totallist=[]
    listtoprocess=df_t['URLs'].str.split(':-:').tolist()
    for j in listtoprocess:
        j = [None if i=='null;' or i=='' else i for i in j]
        j=j+[None]*(11-len(j))
        totallist.append(j)
    df=pd.DataFrame(totallist)
    df_t=pd.concat([df_t,hash_encode_column(df)], axis=1)

    return df_t

    
def count_url(df_t):
    df_t['URLs']=df_t.URLs.str.split(' ').str.len() 
    return df_t


columnlist=[]
for i in range(512):
    columnlist.append('usernamehash_col'+str(i))
columnlist+=['Timestamp','#Followers','#Friends','Retweets',"#Favourites",'Mentions_count',"Hashtag_counts",'URL_counts']
for i in range(41):
    columnlist.append('Entities_embeddings'+str(i))
for i in range(2):
    columnlist.append('Sentiments'+str(i))
for i in range(64):
    columnlist.append('Mentionshash_col'+str(i))
for i in range(64):
    columnlist.append('Hashtagshash_col'+str(i))
for i in range(64):
    columnlist.append('URLshash_col'+str(i))



df=pd.read_feather('clean.ftr')

df1=hash_encode_username(df)
df2=clean_timestamp(df1)
df3=create_entitylist(df2)
df4= split_sentiment(df3)
df5=clean_general(df4,"Mentions")   
df6=count_general(df5,'Mentions')
df7=clean_general(df6,"Hashtags")   
df8=count_general(df7,'Hashtags')
df9=clean_url(8)
df10=count_url(df9)
        
    
        