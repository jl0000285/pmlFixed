# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:01:49 2016

@author: o-4

Simple file parser program written to make data sets with catagorical data in them readible for shogun
machine learning algorithms
"""
import subprocess
import csv
import numpy as np
import itertools
import math
import fparser as fp
import re
import pdb

COMMA_DL=0
SPACE_DL=1
LC_SINGLE=1
LC_SEPERATE=2
LC_SVM_A=3
LC_SVM_B=4
LC_SVM_C = 5
LC_FILE_NAMES=[]
TP_TRUE = True
TP_FALSE = False

class Parser():
    """Parser class"""
    COMMA_DL=0
    SPACE_DL=1
    LC_SINGLE=1
    LC_SEPERATE=2
    LC_SVM_A=3
    LC_SVM_B=4
    LC_SVM_C = 5
    LC_FILE_NAMES=[]
    TP_TRUE = True
    TP_FALSE = False

    def __init__ (self,last_column, file_path, test_par=False, per=0, del_type=','):
       """Create Parser object for given file_path

        Parameters
        ----------
       last column : variable that tells the parser how to treat the last column of data within a given file
           last column values:
                 1. Default (all data in one file) 
                 2. Seperate last column
                 3. Modfify lasts column values to be 1 or -1
                 4. Modify lasts column values to be -1 or +1
       file_path :  name of the target file
       test partition : boolean to tell whether or not to partition a percentage of data set for testing
       per : percent of data set to withhold for testing
       del_type : delimeter to expect within input file"""
       self.last_column = last_column
       self.file_path = file_path
       self.test_par = test_par
       self.per = per
       self.del_type = del_type
       self.filename, self.dirpath, self.ext = self.get_fileinfo()

    def get_fileinfo(self):
        """Parse filename from filename path
        """
        comp = re.compile(r"""
        (.*)   #full directory path containing file
        ([/])  #slash at the end
        (\w*)  #filename
        ([.])  #trailing dot
        (\w*)  #file type
        """,re.VERBOSE)

        comp2 = re.compile(r"""
        (.*)   #full directory path containing file
        ([/])  #slash at the end 
        (\w*?([-]\w*))+ #filename that may or may not contain dash
        ([.]) #trailing dot
        (\w*) #file type
        """,re.VERBOSE)
        try:
            split =  filter(None,comp.split(self.file_path))
            print("First split: {}".format(split))
            filename = split[2]
            dirpath = split[0]
            ext = split[4]
            
        except Exception as ex:
            print("First expression did not work")
            split2 = filter(None,comp2.split(self.file_path))
            print("Second split: {}".format(split2))
            filename = split2[2]
            dirpath = split2[0]
            ext = split2[5]
            
        return filename, dirpath, ext

    def detect_type(self):
        """
        Configure delemeter and last column options based off the file type 
        """
        pass

    def convert_file(self):
        """Read in target file and convert it into a form readable by a learner
        """
        #file_path = "../data/uci/adult/adult.data"
        target_input = []
        """ Conversion boolean-boolean to see if a data set requires set conversion
        i.e does the file contain string data points
        """
        c_b = False

        """
        Check for null byte
        """
        if '\0' in open(self.file_path).read():
            nullByte = True
        else:
            nullByte = False 
        
        #pdb.set_trace()
        with open (self.file_path, 'rb') as csvfile:
            if not nullByte:
                rdr = csv.reader(csvfile, delimiter=self.del_type)
            else:
                rdr = csv.reader((x.replace('\0','') for x in csvfile), delimiter=self.del_type )
            for row in rdr:
                target_input.append(row)
                for dpoint in row:
                    try:
                        float(dpoint)
                    except ValueError:
                        c_b = True;
                        
        """ Clear out empty elements
        """
        target_input = [x for x in target_input if x!=[]]
        
        if c_b == False:
            target_input = [[float(x) for x in r] for r in target_input]

        """
        If conversion is neccessary, iterate thru entire data set and
        add unique values in columns were conversion fails into a list
        for that column.
        """
        cols = []
        colset = set()
        if(c_b == True):
            """
            Perform initial conversion of potential float string objects into actual floats
            """
            for counterA, row in enumerate(target_input):
                #print 'Current i: '+ str(i) + '\n'
                for counterB, dpoint in enumerate(row):
                    #print 'Current j: ' +str(j) + '\n'
                    try:
                        if dpoint != [] and dpoint != None:
                            float(dpoint)
                            #print 'Current nums: ' + str(numA) + ' ' + str(numB) + '\n'
                            target_input[counterA][counterB] = float(dpoint)
                    except ValueError:
                        continue
            #pdb.set_trace()
            #print target_input

            for row in target_input:
                for colcount, dpoint in enumerate(row):
                    try:
                        float(dpoint)
                    except ValueError:
                        if colcount not in colset:
                            colset.add(colcount)
                            cols.append(colcount)
                            colcode = "col_" + str(colcount) + " = [] "
                            exec colcode
             #pdb.set_trace()
             #for name in vars().keys():
             #  print(name)
             #print cols
            for row in target_input:
                for num, dpoint in enumerate(row):
                    if dpoint != [] and dpoint != None:
                        if num in cols:
                            #if j[num] not in col_num
                            #col_num.append(j[num])
                            colcheck = "if row[" + str(num) + "] not in col_" + str(num) + ": \r \t \t"
                            coladd = "col_" + str(num) + ".append(row[" + str(num) + "])"
                            colcom = colcheck + coladd
                            exec colcom
            #pdb.set_trace()
            """
            Once the unique value lists have been crafted,
            replace string values with index of value within
            a given lists in the target_input data structure
            """
            
            for num, row in enumerate(target_input):
                for col in cols:
                    if row != [] and row != None:
                        #target_input[num][i] = col_i.index(target_input[num][i])
                       
                        swapcode = "target_input[num][col] = col_{}.index(target_input[num][col])".format(str(col)) 
                        
                        try:
                            exec swapcode
                        except Exception as ex:
                            pdb.set_trace()
                            pass

        return target_input

    def limit_size(self,target_input,limit=2000):
        """Limit size of data set to given limit """
        import random
        if limit < len(target_input):
            repl_input = []
            for i in range(limit):
                inx = random.randint(0,limit)
                repl_input.append(target_input[inx])
            return repl_input
        else:
            return target_input
        
    def write_csv(self, target_input):
        """ CSV writing handler method. 
        """
        #Prepare target_input for writing
        if self.test_par==True:
            target_input_A,target_input_B = self.partition(target_input, self.per)


        if(self.last_column == Parser.LC_SINGLE):
            if self.test_par==True:
               self.__write_LC_SINGLE__(target_input_A, '.pardataA')
               self.__write_LC_SINGLE__(target_input_B, '.pardataB')
            else:
               self.__write_LC_SINGLE__(target_input, '.parse')

        if(self.last_column == Parser.LC_SEPERATE):
            if self.test_par==True:
               self.__write_LC_SEPERATE__(target_input_A, '.sepdataA')
               self.__write_LC_SEPERATE__(target_input_B, '.sepdataB')
            else:
               self.__write_LC_SEPERATE__(target_input, '.sepCol')

        if(self.last_column == Parser.LC_SVM_A):
            if self.test_par==True:
                target_input_A = self.__mod_LC_SVM_A__(target_input_A)
                target_input_B = self.__mod_LC_SVM_A__(target_input_B)
                self.__write_LC_SEPERATE__(target_input_A, '.svmAtrain')
                self.__write_LC_SEPERATE__(target_input_B, '.svmAtest')
            else:
                target_input = self.__mod_LC_SVM_A__(target_input)
                self.__write_LC_SEPERATE__(target_input, '.svmAall')

        if(self.last_column == Parser.LC_SVM_B):
            if self.test_par==True:
                target_input_A = self.__mod_LC_SVM_B__(target_input_A)
                target_input_B = self.__mod_LC_SVM_B__(target_input_B)
                self.__write_LC_SEPERATE__(target_input_A, '.svmBtrain')
                self.__write_LC_SEPERATE__(target_input_B, '.svmBtest')
            else:
                target_input = self.__mod_LC_SVM_B__(target_input)
                self.__write_LC_SEPERATE__(target_input, '.svmBall')

        if(self.last_column == Parser.LC_SVM_C):
            if self.test_par==True:
                target_input_A = self.__mod_LC_SVM_C__(target_input_A)
                target_input_B = self.__mod_LC_SVM_C__(target_input_B)
                self.__write_LC_SEPERATE__(target_input_A, '.svmCtrain')
                self.__write_LC_SEPERATE__(target_input_B, '.svmCtest')
            else:
                target_input = self.__mod_LC_SVM_C__(target_input)
                self.__write_LC_SEPERATE__(target_input, '.svmCall')
        if self.test_par == True:
            return target_input_A,target_input_B
        else:
            return target_input
    #Static parse, 1 to 1 translation

    def split_last_column(self,target_input):
        """Takes as input a preparsed training or testing set and returns it with
        the last column split off on its own i.e a seperation of input and output
        """
        X = []
        y = []
        row_length = len(target_input[0])
        try: 
            for item in target_input:
                X.append(item[:row_length-2])
                y.append(item[row_length-1])
        except Exception as ex:
            print "There was an exception while spliting last column, returning null arrays"
        return X,y

    def partition(self,target_input, per=0.25):
        """Method to partition data set into
           training and testing set
        """
        ps = math.floor(per*len(target_input))
        target_input_A = []
        target_input_B = []
        for tc, i in enumerate(target_input):
             if tc > ps:
                 target_input_A.append(i)
             if tc < ps:
                 target_input_B.append(i)
        return(target_input_A,target_input_B)

    def __stringify_set__(self, target_input):
        """Convert data set into string
        """
        ti = 0
        for i in target_input:
          tj = 0
          for j in i:
              target_input[ti][tj] = str(tj+1) + ':' + str(target_input[ti][tj])
              tj = tj + 1
          ti = ti + 1

        tc = 0
        print(len(target_input))
        print(len(t_lb))

    def __labels_on_front__(self, target_input):
        """Move sample labels column to front of data set
        """
        pass

    def __write_LC_SINGLE__(self, target_input, ext_name):
        """Write target_input out in one file
        """
        newfile = "{}/{}{}".format(self.dirpath,self.filename,ext_name)
        writefile = open(newfile,'wb')
        wr = csv.writer(writefile, delimiter = self.del_type)
        wr.writerows(target_input)
        return(target_input)

    def __write_LC_SEPERATE__(self, target_input, ext_name):
        """ Write target input into two data sets and last column
        """
        t_lc = []     #target last column
        end = len(target_input[0]) - 1
        for tc, row in enumerate(target_input):
              if row != [] and row != None:
                  t_lc.append([row[end]])
                  del target_input[tc][end]

        newfile  = "{}/{}{}".format(self.dirpath,self.filename,ext_name)
        newfile2 = "{}/{}{}".format(self.dirpath,self.filename,'.plabel')
        w1 = open(newfile,'wb')
        w2 = open(newfile2, 'wb')
        wr1 = csv.writer(w1, delimiter = self.del_type )
        wr2 = csv.writer(w2, delimiter = self.del_type)
        wr1.writerows(target_input)
        wr2.writerows(t_lc)
        return (target_input, t_lc)

    def __mod_LC_SVM_A__(self, target_input):
        """Write svm with two class labeling, -1 and +1
        """
        t_lc = [item[len(item)-1] for item in target_input]
        tb_lc = list(set(t_lc))
        ta_b = [1, -1]
        end = len(target_input[0])-1
        for i,item in enumerate(t_lc):
          if item != [] and item != None:
                ind = tb_lc.index(item)
                target_input[i][end] = ta_b[ind]
        return(target_input)

    def __mod_LC_SVM_B__(self, target_input):
        """.svm format with -1 +1 binary labels
           sending files that are not two class will result in an out
           of bounds error with t_lb
        """
        t_lc = [item[len(item)-1] for item in target_input]
        tb_lc = list(set(t_lc))
        ta_b = [1.0, -1.0]
        end = len(target_input[0])-1
        for i, item in enumerate(t_lc):
          if item != [] and item != None:
                ind = tb_lc.index(item)
                target_input[i][end] = ta_b[ind]
        return (target_input)

    def __mod_LC_SVM_C__(self, target_input):
        """Write svm a format on last column
        """  #.svm format
        t_lc = []
        end = len(target_input[0]) - 1
        for tc, row in enumerate(target_input):
           if row != [] and row != None:
              t_lc.append([row[end]])
              del target_input[tc][end]
        for ti, row in enumerate(target_input):
            for tj, datap in enumerate(row):
              target_input[ti][tj] = str(tj+1) + ':' + str(target_input[ti][tj])

        for tc, row in enumerate(target_input):
           if row != [] and row != None:
              target_input[tc].insert(0,str(t_lc[tc][0]))
        return (target_input)
