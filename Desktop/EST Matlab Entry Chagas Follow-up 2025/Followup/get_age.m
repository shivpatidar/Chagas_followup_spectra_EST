function age=get_age(header)
        header=strsplit(header,'\n');
        age_tmp=header(startsWith(header,'# Age:'));
        if isempty(age_tmp)
            age=NaN;
        else
            age_tmp=strsplit(age_tmp{1},':');
            age=str2double(age_tmp{2});
        end
    end