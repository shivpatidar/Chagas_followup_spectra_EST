function weight=get_weight(header)
        header=strsplit(header,'\n');
        weight_tmp=header(startsWith(header,'# Weight:'));
        if isempty(weight_tmp)
            weight=NaN;
        else
            weight_tmp=strsplit(weight_tmp{1},':');
            weight=str2double(weight_tmp{2});
        end
    end