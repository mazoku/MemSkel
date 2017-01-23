function c = load_spline(path)
% reads a .shapes file specified with a given path
c = cell(1,1);
fid = fopen(path,'r');

textscan(fid,'%s',4,'delimiter', '\n');
textscan(fid,'%s',7,'delimiter', '\n');
i = 1;
while (~feof(fid))
    sliceNo=textscan(fid,'in slice=%f');
    x = textscan(fid,'Supercolors=%s');
    x{1}
    sliceNo{1}
    c(i,1) = sliceNo;
    p = textscan(fid, '%f','delimiter',',');
    c{i,2} = reshape(p{1},[length(p{1})/2 2])';
    textscan(fid,'%s',6,'delimiter', '\n');
    i = i+1;
end

fclose(fid);