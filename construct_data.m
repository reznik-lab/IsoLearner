% Iterate through the different metabolites in the list, appending the
% isotopolouge breakdown for each to an ultimate data table!
file_name = input("Enter file name: ", "s")

msi = getappdata(handles.figure1, 'msi');
pks = getappdata(handles.figure1, 'pks');
num_points = size(msi.data, 2);

% Need to automate this later
data = zeros(num_points,1);
num_metabolites = 353; %13;
all_isotopolouges = [];

for i = 1:num_metabolites
    mypk = Mzpk(pks.sdata(i));  % select a peak
    metabolite_name = mypk.name;
    num_isotopolouges = mypk.maxM_;
    msi_temp = msi_get_idata(msi,mypk); % get idata
    msi_temp = msi_get_isoidata(msi_temp,mypk); % get idata for isotopologue
    % msi_temp.isoidata.idata
    isotopolouge_names = [];
    if mod(i,25) == 0;
        i
    end
    for j = 0:num_isotopolouges
        %x = strcat(metabolite_name, ' m+', j)
        x = strcat(metabolite_name, sprintf(" m+%02d",j));
        isotopolouge_names{end+1} = x;
    end
    all_isotopolouges = horzcat(all_isotopolouges, isotopolouge_names);
    data = horzcat(data, msi_temp.isoidata.idata);
end

data(:,1) = [];
all_isotopolouges = string(all_isotopolouges);
coords = ["x", "y"];
all_isotopolouges = horzcat(coords, all_isotopolouges);

x_coord = [msi.data(1:end).x].';
y_coord = [msi.data(1:end).y].';

final = horzcat(y_coord, data);
final = horzcat(x_coord, final);
final = vertcat(all_isotopolouges, final);

writematrix(final,sprintf('generated-data/brain-m0-no-log/Brain-15NNH4Cl/B15NNH4Cl-FML-%s.csv', file_name));
clear;
