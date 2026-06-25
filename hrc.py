def hrc(A,E,L,index):
# % This function calculates the 'Reconstructed Harmonic Components' using the
# % DAHD modes (E) and DAHD coefficients (A) corresponding to a given frequency band. 
# % Both of these matrices may have had columns trimmed but they must
# % have the same number of columns. 
# %
# % Input:  A - DAHD coefficients (DAHC) matrix (N-M'+1 by k)
# %         E - DAHD mode (DAHM) matrix (L*M' by k)
# %         L - number of channels in the original data matrix.
# %     index - (optional) index vector into A and E. For example, if 
# %             index = [1 2], the HRC for combined modes 1 and 2 will
# %             be computed. Since DAHMs are paired, k should be even number.
# %
# %
# % Output: Rs - Matrix of HRC for combination of input DAHMs/DAHCs, N by L.
# %
# %
# %  See Appendix A of Chekroun, M. D., and D. Kondrashov, 2017:
# %  Data-adaptive harmonic spectra and multilayer Stuart-Landau models,
# %  Chaos, 27, 093110: doi:10.1063/1.4989400
# %  and 
# %  Kondrashov D, Ryzhov EA, Berloff P. Data-adaptive harmonic analysis of oceanic waves and turbulent flows. 
# %  Chaos, 30, doi: 10.1063/5.0012077
# %  Written by Dmitri Kondrashov.   Version date 1/29/25
# %  Please send comments and suggestions to dkondras@atmos.ucla.edu

    ml,k=E.shape
    ra, ka=A.shape
# if k~=ka, error('E and A must have the same number of columns.'), end
# if nargin==3, index=1:k; end

# M=ml/L;   %     These lines assume that E and A
# N=ra+M-1; %     have the "right" row dimensions.
# MT=ra;

# R=zeros(N,L*length(index));
# Z=zeros(M-1,k);
# A=[A' Z']; % zero pad A to make it N by k
# A=A';
# Ej=zeros(M,L);

# % Calculate HRCs
# for j=1:length(index)
#   Ej(:)=E(:,index(j)); % Convert the j-th DAHM into a matrix of filters
#   for i=1:L     % Compute the HRCs for the j-th DAHM/DAHC pair
#     R(:,(j-1)*L+i)=filter(Ej(:,i),M,A(:,index(j)));
#   end
# end
# % Adjust first M-1 rows and last M-1 rows
# Rs = zeros(N,L);
# for j=1:length(index)
# Rs = Rs+R(:,(j-1)*L+1:j*L);
# end
#        for idat = 1: MT 
# 	if idat < M
# %
# % .. 1 <= i < M
# %
#          Rs(idat,:)= Rs(idat,:)*M/idat;

# 	end
# 	end
# %
# %
# % .. N-M+1 < i <= N
# %    --------------
# 	for idat=MT+1:N

#        if idat <= M 
# %
# % .. 1 <= i < M     (this section only for short timeseries: N < 2M)
# %
#          Rs(idat,:) = Rs(idat,:)*M/MT; 

#        else
# %
# % .. M <= i <= N-M+1
# %
#          Rs(idat,:) = Rs(idat,:)*M/(N-idat+1); 
#        end
# 	end


# return
