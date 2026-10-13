
from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""

// Add (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    for (uint32_t i=0;i<${size};i++){
    %if detect_positive:
        %if detect_negative:
            ${Y}[i] = (isinf(${X}[i]) != 0); // Note: IsInf is sign preserving!
        %else:
            ${Y}[i] = (${X}[i] == INFINITY); 
        %endif
    %elif detect_negative:
        ${Y}[i] = (${X}[i] == -INFINITY); 
    %else:
        ${Y}[i] = 1;
    %endif
    }
END_SINGLE_CORE
""")
