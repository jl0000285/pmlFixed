from fparser import parser
set = parser(1,'/home/o-4/Downloads/meta/pml/data/init/adult/adult.data')
target_input = set.convert_file()
set.write_csv(target_input)
