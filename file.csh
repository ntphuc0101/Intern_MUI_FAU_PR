#!/bin/csh
# Example: 4
foreach x (`find ./ -name "*image*"`)
     set a = `echo "$x" | cut -d "/" -f2`
     set b = `echo "$x" | cut -d "/" -f3`	
     echo ${a}_${b}
     set dir = `pwd`
     echo "save file as " + ${dir}/${a}_${b}.png 
     cp $x ${dir}/${a}_${b}.png
end
#EOF
