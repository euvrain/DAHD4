
            function [fE2,VP,FEP]=DAHD4freq_part_weight(X,W,NFE,NP,wt)
% This function calculates DAHD spectrum in frequency-domain formulation
% Input: input data array X (N x L) 
%        W - embedding window M
%        NFE - number of frequencies (maximum is M)
%        NP  - number of spectral elements per frequency (maximum is L)
%        wt - method for weignting temporal correlations
% Output: 
%         array  FEP - Hermitian DAHD eigenvectors sorted by frequencies, max size L x L x M
%         array  VP -  Hermitian DAHD eigenvalues sorted by frequencies, max size L x M
%         array  fE2 - resolved frequency bins, size 1 x M
%  See Kondrashov D., et al. 2020 Data-adaptive harmonic analysis of oceanic waves and turbulent flows. 
%  Chaos, 30, doi: 10.1063/5.0012077 
%  and 
%  Kondrashov D., et al. 2026 Accurate and robust real-time prediction of September Arctic sea ice 
%  Chaos: 36, doi: 10.1063/5.0295634
%  Written by Dmitri Kondrashov.   Version date 6/22/26
%  Please send comments and suggestions to dkondras@atmos.ucla.edu


            WW=2*W-1;
            dim=size(X,2);
            cspec=zeros(WW,dim,dim);

            if strcmp(wt, 'hamming')
    weight = hamming(WW, 'symmetric');
elseif strcmp(wt, 'hanning')
    weight = hanning(WW, 'symmetric');
elseif strcmp(wt, 'blackman')
    weight = blackman(WW, 'symmetric');
elseif strcmp(wt, 'hann')
    weight = hann(WW, 'symmetric');
elseif strcmp(wt, 'bartlett')
    weight = bartlett(WW);
else
    weight = ones(WW, 1);
end



            method='unbiased';
            for k=1:dim
            for l=1:dim
            cspec(:,k,l)=weight.*xcorr(X(:,l),X(:,k),W-1,method);
            end
            end



            toto=fft(cspec);

            VP=zeros(NP,NFE);
            VPT=zeros(NP,NFE);
            FEP=zeros(dim,NP,NFE);
            for NF=1:NFE
            cf2=exp(-i*(NF-1)*pi/WW);
            toto2=squeeze(toto(NF,:,:)*cf2);
            if trace(toto2)<0 toto2=-toto2;
            end

            [ee,dd]=eigs(toto2,NP);
            VPT(:,NF)=real(diag(dd));
            VP(:,NF)=abs(diag(dd));
            FEP(:,:,NF)=conj(ee);

            numPositive = sum(diag(dd) > 0);      
            numNegative = sum(diag(dd) < 0); 
            display([num2str(NF) ' >0:' num2str(numPositive) ' <0:' num2str(numNegative)]);
%%%%%%%%%%%%%%%%%%%%e
            end

            M1=2*W-1;

            MM2=(M1-1)/2;
            for j=1:NFE
            fE2(j)=0.5*(j-1)/MM2;
            end

          

            return

