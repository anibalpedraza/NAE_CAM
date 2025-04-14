function calculateStatistics(dataPath,outputPath)

    % Explore main path and list available networks
    networkList=dir(dataPath);
    networkList=networkList(3:end);
    
    % Define some global parameters
    metrics={'mediaIntensidadPixeles','varianza'};% 'ssim'
    combinationPairs=[1,2;1,3;1,4;2,3;2,4]; % 1,2;1,3
    statsTableFull=table();
    
    % Iterate over networks (only directories)
    for i=1:length(networkList)
        if networkList(i).isdir
            networkName=networkList(i).name;
            networkPath=fullfile(dataPath,networkName);
            fprintf(['Processing network: ',networkName,'\n']);
            % Iterate over metrics
            for j=1:length(metrics)
                metricName=metrics{j};
                metricFilePath=fullfile(networkPath,[networkName,'_',metricName,'.csv']);
                fprintf(['\t','Processing metric: ',metricName,'\n']);
                % Read data
                tableData=readtable(metricFilePath,'ReadRowNames',true,'Delimiter',...
                    ';','DecimalSeparator',',');
                nVars=size(tableData,2);
                % Define names of combinationPairs
                datasetNames=tableData.Properties.RowNames;
                combinationsNames=cell(size(combinationPairs,1),1);
                for k=1:size(combinationPairs,1)
                    combinationsNames{k}=[datasetNames{combinationPairs(k,1)},'_vs_',datasetNames{combinationPairs(k,2)}];
                end
                % Empy table to store ttest results
                statsTable=table('Size',[size(combinationPairs,1),7],'VariableTypes',...
                    {'string','string','string','double','double','double','double'},...
                    'VariableNames',{'network','metric','combination','ttestT','ttestP','utestU','utestP'});
                % Iterate over combinationPairs
                for c=1:size(combinationPairs,1)
                    fprintf(['\t','\t','Processing combination: ',combinationsNames{c},'\n']);
                    % Plot with legend and no graphics
                    figure('Visible','off');
                    scatter(1:nVars,tableData{combinationPairs(c,1),:},'blue');
                    hold on;
                    scatter(1:nVars,tableData{combinationPairs(c,2),:},'red');
                    legend(datasetNames{combinationPairs(c,1)},datasetNames{combinationPairs(c,2)});
                    xlabel('Number of samples');
                    ylabel('Mean intensity value');
                    % Save plot
                    plotPath=fullfile(outputPath,[networkName,'_',metricName,'_',...
                        combinationsNames{c},'.png']);
                    % Position: left,down,right,up
                    set(gca,'position',[0.08 0.10 0.89 0.88]);
                    saveas(gcf,plotPath);
                    % T-Test
                    [ttestH,ttestP,ttestCI,ttestStats]=ttest2(tableData{combinationPairs(c,1),:},tableData{combinationPairs(c,2),:});
                    % U-Test Mann-Whitney - Wilcoxon ranksum
                    [utestP,utestH,utestStats]=ranksum(tableData{combinationPairs(c,1),:},tableData{combinationPairs(c,2),:});
                    % Store results in the table
                    statsTable{c,'network'}=string(networkName);
                    statsTable{c,'metric'}=string(metricName);
                    statsTable{c,'combination'}=string(combinationsNames{c});
                    statsTable{c,'ttestT'}=round(ttestStats.tstat,2); % ttestT
                    statsTable{c,'ttestP'}=round(ttestP,2);
                    statsTable{c,'utestU'}=round(utestStats.ranksum,2); % ttestT
                    statsTable{c,'utestP'}=round(utestP,2);
                end
                statsTableFull=[statsTableFull;statsTable];
            end
        end
    end
    
    % Write global ttest results to file
    statsPathFull=fullfile(outputPath,'statsResultsFull.csv');
    writetable(statsTableFull,statsPathFull,'Delimiter',',');
    
    fprintf('Finished processing all networks and metrics\n');

end