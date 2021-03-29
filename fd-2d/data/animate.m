close all;

%results; 

for ind = 1:32
    
    newFile = fullfile("results"+num2str(ind)+".m");
    run(newFile);
    
    surf(F);
    % axis([-1 1 -1 1, -0.06, 0.06]);
    caxis([-0.04, 0.04]); 
    colorbar;
    shading interp;
    pause(0.1);
end
    