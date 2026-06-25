
            function [EVP]=DAHM4_ex(vv,W,NF,NT)
%  this function computes EVP - space-time DAHMs obtained 
%  by an inverse Fourier transform of Hermitian DAHD eigenvectors vv
%  W -  embedding window
%  NF - frequency index 
%  NT - number of DAHMs to compute
%
%  See Kondrashov D et al. 2020 Data-adaptive harmonic analysis of oceanic waves and turbulent flows. 
%  Chaos, 30, doi: 10.1063/5.0012077 
%  and 
%  Kondrashov D., et al. 2026 Accurate and robust real-time prediction of September Arctic sea ice 
%  Chaos: 36, doi: 10.1063/5.0295634
%  Written by Dmitri Kondrashov.   Version date 6/22/26
%  Please send comments and suggestions to dkondras@atmos.ucla.edu           
 
            D=size(vv,1);

            EVP=zeros((2*W-1)*D,NT);

            fftv=zeros(2*W-1,D,NT);
           if NF > 1
           for i=1:NT/2
           i2=2*i-1;
           cvv=sqrt(-1)*vv(:,i);
           fftv(NF,:,i2)=sqrt(W-0.5)*vv(:,i);
           fftv(end-NF+2,:,i2)=sqrt(W-0.5)*conj(vv(:,i));
           fftv(NF,:,i2+1)=sqrt(W-0.5)*cvv;
           fftv(end-NF+2,:,i2+1)=sqrt(W-0.5)*conj(cvv);
           end
           else
           for i=1:NT/2
           cvv=sqrt(-1)*vv(:,i);
           fftv(NF,:,i)=sqrt(2*W-1)*vv(:,i);
           end
           end
           Evv=ifft(fftv,[],1);
           EVP=reshape(Evv,size(Evv,1)*size(Evv,2),size(Evv,3));



           return EVP %fix: empty return statement

