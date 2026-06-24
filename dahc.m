  function [A]=dahc(X,E)
%        Syntax: [A]=dahc(X,E);
% This function calculates the 'DAH coefficients' of the data
% matrix X (N x L) from the eigenfunction matrix E (from DAHD function).
% The eigenvector matrix may be of reduced size (k columns) - one
% DAHC will be calculated for each supplied eigenvector.
%
% Output: A - DAH coeffcients matrix (N-M+1 x k)
%
%  See Appendix A of Chekroun, M. D., and D. Kondrashov, 2017:
%  Data-adaptive harmonic spectra and multilayer Stuart-Landau models,
%  Chaos, 27, 093110: doi:10.1063/1.4989400
%  Written by Dmitri Kondrashov.   Version date 1/13/18
%  Please send comments and suggestions to dkondras@atmos.ucla.edu

[N,L]=size(X);
[M,k]=size(E);                
M=M/L; % Calculate the *actual* value for M
if k>L*M, error('Improper specification of k.'), end
A=zeros(N,k);
a=zeros(N,L);
Ej=zeros(M,L);

for K=1:k
  Ej(:)=E(:,K);
  Ej=flipud(Ej);
  for j=1:L
    a(:,j)=filter(Ej(:,j),1,X(:,j));
  end
  if L>1
    A(:,K)=sum(a')';
  else
    A(:,K)=a;
  end
end

A=A(M:N,:);

