#!/bin/csh
# Example: 4
foreach x (`ls -d */`)
     echo "working DIR" + $x
     cp  *csh $x
     cd $x
     ./file.csh
     ./remove.csh
     cd ..
end
#EOF
