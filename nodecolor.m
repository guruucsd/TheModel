function [color]=nodecolor(newExpertName,n)

if strcmp(newExpertName,'Faces')
    color=[(n/10) n/10 1-n/10];
else
    if strcmp(newExpertName,'Butterflies')
        if n>10
            color=[1-((n-10)/10) (n-10)/10 (n-10)/10];
        else
            color=[(n/10) n/10 1-n/10];
        end
    elseif strcmp(newExpertName,'Cars')
        if n>10
            color=[1-((n-10)/10) (n-10)/10 (n-10)/10];
        else
            color=[(n/10) n/10 1-n/10];
        end
    elseif strcmp(newExpertName,'Leaves')
        if n>10
            color=[1-((n-10)/10) (n-10)/10 (n-10)/10];
        else
            color=[(n/10) n/10 1-n/10];
        end
    else
        if n>10
            color=[1-((n-10)/10) (n-10)/10 (n-10)/10];
        else
            color=[(n/10) n/10 1-n/10];
        end
    end
end


end
