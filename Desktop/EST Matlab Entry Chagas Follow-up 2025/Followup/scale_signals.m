function signals=scale_signals(signals,header)

header=strsplit(header,'\n');

for j=1:size(signals,2)

    header_tmp=header{1+j};
    header_tmp=strsplit(header_tmp,' ');
    header_tmp=header_tmp{contains(header_tmp,'/mV')};

    baseline=extractBetween(header_tmp,'(',')');
    baseline=str2num(baseline{1});

    gain=extractBefore(header_tmp,'(');
    gain=str2num(gain);

    signals(:,j)=(signals(:,j)-baseline)/gain;


end
