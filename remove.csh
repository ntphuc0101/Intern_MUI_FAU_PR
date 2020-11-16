#!/bin/csh
# Example: 4
foreach x (`ls -d */`)
     rm -rf $x
 
end
#EOF
