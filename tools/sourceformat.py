from io import StringIO, BytesIO
import pymarc
import requests
import string
import pandas as pd
import tarfile
try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

#metadata for htrc worksets
def htrc(self):
    
    #variables/arrays and stuff
    
    #string of keywords per volume/htid
    keywords = ""
    
    #array of all the keywords per each volume/htid, to add to the file
    keylist = []

    #get htids of the volumes
    htids = self['htid'].values.tolist()
    #iterate through list of htids
    for id in range(len(htids)):
        htid = htids[id]
        
        #api call for the extra metadata using htid
        extradata = requests.get("https://catalog.hathitrust.org/api/volumes/full/htid/"+htid+".json")
        
        #turn the request into a json file
        extradata = extradata.json()

        #get record id and use it to get the xml/marc file with the actual metadata
        recid = extradata['items'][0]['fromRecord']
        xmlmarc = extradata['records'][recid]['marc-xml']

        #turn the formatted xml into an actual pymarc
        xml = StringIO(xmlmarc)
        marc = pymarc.parse_xml_to_array(xml)[0]
        xml.close()

        for term in marc.get_fields('650'):
            if "http" in (term.value()).lower():
                keywords+= ""
            elif "ocolc" in (term.value()).lower():
                keywords+=""
            else:
                keywords+=term.value().translate(str.maketrans('','', string.punctuation))+"; "
        keylist.append(keywords)
    self['Keywords'] = keylist
    return self

def htrcxtra(self):
        
    #variables/arrays and stuff
    
    #string of keywords per volume/htid
    pages = ""
    
    #array of all the keywords per each volume/htid, to add to the file
    pagecount = []

    #get htids of the volumes
    htids = self['htid'].values.tolist()
    #iterate through list of htids
    for id in range(len(htids)):
        htid = htids[id]
        
        #api call for the extra metadata using htid
        extradata = requests.get("https://catalog.hathitrust.org/api/volumes/full/htid/"+htid+".json")
        
        #turn the request into a json file
        extradata = extradata.json()

        #get record id and use it to get the xml/marc file with the actual metadata
        recid = extradata['items'][0]['fromRecord']
        xmlmarc = extradata['records'][recid]['marc-xml']

        #turn the formatted xml into an actual pymarc
        xml = StringIO(xmlmarc)
        marc = pymarc.parse_xml_to_array(xml)[0]
        xml.close()

    for term in marc.get_fields('350'):
        pages+=term.value()
    pagecount.append(pages)
    self['pages'] = pagecount
    return self


#format files from dimensions
def dim(file):
    formatted = file.drop(file.columns[[0]],axis=1)

    done = pd.read_csv(StringIO((formatted.to_csv(header=False,index=False))))
  
    return done



def readPub(tar):

    #list to put xmls from tarfile in
    xmllist = []

    readfile = BytesIO(tar)

    #get the files from the tarfile into the list
    files = tarfile.open(fileobj=readfile, mode = 'r:gz', )
    for member in files.getmembers():
        singlefile = files.extractfile(member)
        if singlefile is not None:
            article = singlefile.read()
            article = article.decode("utf-8")
            article = StringIO(article)
            xmllist.append(article)

    #lists for each data point
    titles = []
    years = []
    keys = []
    authors = []
    publishers = []
    journaltitles = []
    
    #go through each xml file in the list
    for art in range(len(xmllist)):

        #make a parseable element tree out of the xml file
        tree = ET.parse(xmllist[art])
        root = tree.getroot()

        #remove parts of the main branch that do not have metadata that we care about
        for child in list(root):
            if(child.tag!="front"):
                root.remove(child)

        #names to concatnate for each article
        firstname = []
        lastname = []

        #individual strings for multiple keywords/titles
        key = ""
        title = ""
        

        for target in root.iter('article-title'):
            if target.text is not None:
                title += target.text + ", "
            else:
                title += " "
        for target in root.iter('kwd'):
            if target.text is not None:
                key+=target.text+ "; "
            else:
                key += " "
        for target in root.iter('year'):
            year=int(target.text)
            years.append(year)
        for names in root.iter('given-names'):
            firstname.append(names.text)
        for names in root.iter('surname'):
            lastname.append(names.text)
        for target in root.iter('journal-title'):
            jtitle = target.text
            journaltitles.append(jtitle)
        for target in root.iter('publisher-name'):
            publisher = target.text
            publishers.append(publisher)

        titles.append(title)
        keys.append(key)

        fullnames = [first + ' ' + last for first, last in zip(firstname,lastname)]

        #join the names into a single string with authors
        author = str.join(', ', fullnames)

        authors.append(author)

    data = pd.DataFrame()

    data["Title"] = pd.Series(titles)
    data["Keywords"] = pd.Series(keys)
    data["Authors"] = pd.Series(authors)
    data["Year"] = pd.Series(years)
    data["Document Type"] = pd.Series(publisher)
    data["Source title"] = pd.Series(journaltitles)

    data.fillna(value = "empty", inplace = True)

    return data


def readxml(file):
    root = ET.fromstring(file)



    #remove stuff from the xml that we do not need
    for child in list(root):
        for lchild in list(child):
            if(lchild.tag!="front"):
                child.remove(lchild)

    #get stuff

    keys = []
    titles = []
    authors = []
    jtitle = []
    publishers = []
    years = []

    for child in list(root):
        for article in list(child):
            key = ""
            firstname = []
            lastname = []
            for target in article.iter('article-title'):
                
                if target.text is not None:
                    titles.append(target.text)
                else:
                    titles.append("empty")
            for target in article.iter('kwd'):
                if target.text is not None:
                    key+= target.text + "; "
                else:
                    key += ""
            keys.append(key)
            for target in article.iter('given-names'):
                firstname.append(target.text)
            for target in article.iter('surname'):
                lastname.append(target.text)
            
            fullnames = [first + ' ' + last for first, last in zip(firstname,lastname)]
            author = str.join(', ', fullnames)
            authors.append(author)

            for target in article.iter('journal-title'):
                jtitle.append(target.text)
            for target in article.iter('publisher-name'):
                publishers.append(target.text)

            for target in article.iter('year'):
                years.append(int(target.text))

    frame = pd.DataFrame()

    frame["Title"] = pd.Series(titles)
    frame["Keywords"] = pd.Series(keys)
    frame["Authors"] = pd.Series(authors)
    frame["Year"] = pd.Series(years)
    frame["Document Type"] = pd.Series(jtitle)
    frame["Source title"] = pd.Series(publishers)

    frame.fillna(value = "empty", inplace = True)

    return frame

def medline(file):

    textfile = file.read()


    text = textfile.decode()





    authors = []
    titles = []
    year = []
    meshkeys = []
    otherkeys = []

    #articles are separated by newlines so seperate them
    articles = text.split('\n\n')

    for paper in articles:
        names = ""
        meshk = ""
        otherk = ""         
        largetext = paper.splitlines()
        for line in largetext:
            #title
            if "TI  - " in line:
                #checking if the title goes over another line, and to add it if it does
                startpos = line.index("-") + 2
                if "- " not in(largetext[largetext.index(line)+1]):
                    titles.append(line[startpos:] +  " " + largetext[largetext.index(line)+1].strip())
                else:
                    titles.append(line[startpos:])
            #author
            if "FAU - " in line:
                startpos = line.index("-") + 2
                names+= line[startpos:] + "; "
            #year
            if "DP  - " in line:
                startpos = line.index("-") + 2
                year.append(int(line[startpos:startpos+4]))
            #key terms
            if "MH  - " in line:
                startpos = line.index("-") + 2
                meshk += line[startpos:] + "; "
            if"OT  - " in line:
                startpos = line.index("-") + 2
                otherk += line[startpos:] + "; "
    
        authors.append(names)
        meshkeys.append(meshk)
        otherkeys.append(otherk)

    frame = pd.DataFrame()
    
    frame['Title'] = pd.Series(titles)
    frame['Authors'] = pd.Series(authors)
    frame['Year'] = pd.Series(year)
    frame['MeSH Keywords'] = pd.Series(meshkeys)
    frame['Other Keywords'] = pd.Series(otherkeys)

    frame.fillna(value = "empty", inplace = True)

    return frame