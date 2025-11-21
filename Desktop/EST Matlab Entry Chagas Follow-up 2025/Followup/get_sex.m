function sex=get_sex(header)
        header=strsplit(header,'\n');
        sex_tmp=header(startsWith(header,'# Sex:'));
        if isempty(sex_tmp)
             sex=NaN;
        else
            sex_tmp=strsplit(sex_tmp{1},':');
            sex_str=strtrim(sex_tmp{2});
            if startsWith(sex_str,'Fem')
                sex=0;
            elseif startsWith(sex_str,'Mal')
                sex=1;
            else
                sex=2; % Unknown/Other
            end
        end
    end
