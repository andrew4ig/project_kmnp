%The code bellow allows you to solve Schrödinger equation via Method of
%variations and Petrubation theory, if one cannot be solved analytically
%Runnig a programm calls a menu, where you can choose type of a task 
%needed to be solved
%For each function there are some determinated types of force fields (or
%petrubation types), if they are not enough you can change it directly in code
%(comments show where you can do it)
%also each function has an ineractive module that helps you to understand
%dependences between variables

%preparing workspace 
%closing all figures
all_fig = findall(0, 'type', 'figure');
close(all_fig)
%deleting all the variables
clear
%clearing command window
clc

%calling the menu
menus;

function menus(~)
%creating a figure, where buttons are located
fig=uifigure('Name','Menu','Resize','off');
fig.Units='normalized';
fig.Position=[0.35 0.3 0.3 0.4];
fig.Units='pixels';

%creating a text space for "petrubation theory" - unpressible button
pnl = uipanel(fig);
pnl.Position=[0.19*fig.Position(3) 0.4*fig.Position(4)  0.62*fig.Position(3) 0.5*fig.Position(4)];
pnl.BackgroundColor=[0.85 0.85 0.85];
Petr=uilabel(fig,'Text','Perturbation theory',...
        'WordWrap','on','HorizontalAlignment','center');
Petr.Position=[0.19*fig.Position(3) 0.8*fig.Position(4)  0.62*fig.Position(3) 0.1*fig.Position(4)];

%creating a text space for "Stationary petrubation theory" - unpressible button
pnl1 = uipanel(fig);
pnl1.Position=[0.195*fig.Position(3) 0.6*fig.Position(4)  0.61*fig.Position(3) 0.2*fig.Position(4)];
pnl1.BackgroundColor=[0.85 0.95 0.95];
Petr=uilabel(fig,'Text','Stationary',...
        'WordWrap','on','HorizontalAlignment','center',...
        'BackgroundColor',[0.9 0.9 0.9]);
Petr.Position=[0.195*fig.Position(3) 0.7*fig.Position(4)  0.61*fig.Position(3) 0.1*fig.Position(4)];

%creating a button for "Degenerated stationary petrubation" - pressible button
Deg=uibutton(fig,'ButtonPushedFcn', @(Deg,fig) Degbutpush(Deg,fig));
Deg.Position=[0.51*fig.Position(3) 0.61*fig.Position(4)  0.28*fig.Position(3) 0.08*fig.Position(4)];
Deg.Text='Degenerated';

%creating a button for "Nongenerated stationary petrubation" - pressible button
Ndeg=uibutton(fig,'ButtonPushedFcn', @(Ndeg,fig) Ndegbutpush(Ndeg,fig));
Ndeg.Position=[0.21*fig.Position(3) 0.61*fig.Position(4)  0.28*fig.Position(3) 0.08*fig.Position(4)];
Ndeg.Text='Non-Degenerated';

%creating a text space for "Nonstationary petrubation theory" - unpressible button
pnl2 = uipanel(fig);
pnl2.Position=[0.195*fig.Position(3) 0.4*fig.Position(4)  0.61*fig.Position(3) 0.2*fig.Position(4)];
pnl2.BackgroundColor=[0.95 0.95 0.85];
Petr=uilabel(fig,'Text','Nonstationary',...
        'WordWrap','on','HorizontalAlignment','center',...
        'BackgroundColor',[0.9 0.9 0.9]);
Petr.Position=[0.195*fig.Position(3) 0.5*fig.Position(4)  0.61*fig.Position(3) 0.1*fig.Position(4)];

%creating a button for "Nonstationary petrubation - Golden rule" - pressible button
Nonst=uibutton(fig,'ButtonPushedFcn', @(Nonst,fig)Nonstbutpush(Nonst,fig));
Nonst.Position=[0.21*fig.Position(3) 0.41*fig.Position(4)  0.58*fig.Position(3) 0.08*fig.Position(4)];
Nonst.Text='Golden Rule';

%creating a panel for "Variation" - pressible button
pnlVar = uipanel(fig);
pnlVar.Position=[0.19*fig.Position(3) 0.1*fig.Position(4)  0.62*fig.Position(3) 0.2*fig.Position(4)];
pnlVar.BackgroundColor=[0.95 0.85 0.95];
Var=uilabel(fig,'Text','Method of variations',...
        'WordWrap','on','HorizontalAlignment','center',...
        'BackgroundColor',[0.85 0.85 0.85]);
Var.Position=[0.19*fig.Position(3) 0.2*fig.Position(4)  0.62*fig.Position(3) 0.1*fig.Position(4)];

%creating a button for "Random Variation" - pressible button
Rand=uibutton(fig,'ButtonPushedFcn', @(Rand,fig) Varbutpush(Rand,fig));
Rand.Position=[0.21*fig.Position(3) 0.11*fig.Position(4)  0.28*fig.Position(3) 0.08*fig.Position(4)];
Rand.Text='Randon method';

%creating a button for "Linear Variation" - pressible button
Line=uibutton(fig,'ButtonPushedFcn', @(Line,fig) Linbutpush(Line,fig));
Line.Position=[0.51*fig.Position(3) 0.11*fig.Position(4)  0.28*fig.Position(3) 0.08*fig.Position(4)];
Line.Text='Linear method';
end

function Linbutpush(~,~)
%creating a figure, where graphics and listbox located
fig=figure('Name','Method of variations: Linear type','NumberTitle','off');
fig.Units='normalized';
fig.Position=[0 0 1 1];

%creating an interactive module, where you can choose type of force field.
%function for realizing method of variations is called here
FeildDefText=uicontrol(fig,'style','text','BackgroundColor',...
                      get(fig, 'color'),'String','Define Feild type',...
                     'FontSize',16,'HorizontalAlignment', 'center');
FeildDefText.Units='normalized';
FeildDefText.Position=[0.15 0.35 0.2 0.1];
FeildType=uicontrol(fig, 'style','listbox',...
                   'string',{'\/-type','/\-type','cos-type','barrier','step-type'},...
                   'FontSize',12,'HorizontalAlignment', 'center',...
                   'Value',1,'Callback',@LinearComb);
FeildType.Units='normalized';
FeildType.Position=[0.15 0.30 0.2 0.1];
end

function LinearComb(h,~)
type=h.Value;
%defining constants
hbar=1.0546e-34;
m0=9.1e-31;
e=1.6e-19;

%forming a task
L=1e-8;
Np=100;
x=linspace(-L/2,L/2, Np);
dx=x(2)-x(1);
koef=-hbar^2/(2*m0*12*(dx^2));

%defining potential feild
switch type
    case 1
        U=abs(x/(L/2)*e/2);
    case 2
        U=e/2-abs(x/(L/2)*e/2);
    case 3
        U=e/2*(1+sin(pi*10*x/L));
    case 4
        U=e/2*(exp(-x.^2/(L/10)^2));
    case 5
        U=heaviside(x)*e/2;
end

%numerical solution for hamiltonian
    %defining secind devirative
E=eye(Np)*(-30);
E=E+diag(ones(1,Np-1)*16,-1);
E=E+diag(ones(1,Np-1)*16,1);
E=E+diag(ones(1,Np-2)*(-1),-2);
E=E+diag(ones(1,Np-2)*(-1),2);

%Hamiltonian
H=E*koef+diag(U);

%finding eigenvalues and eigenvectors
[P,Ei]=eig(H);
Ei=diag(Ei);

%choosing 10 solutions of particle in a box as a basis
Count=10;
m=1:Count;
phi=sqrt(2/L)*sin(pi/2*m+pi*x'*m/L);
phi=phi./sqrt(diag(phi'*phi)');

%solving variation
%declaring matrixes
Hij=phi'*H*phi;
Sij=phi'*phi;
%solving for minimal E
syms E
E=double(solve(det(Hij-E*Sij)==0));
Em=E(1);
%determenating constants
c=null((Hij-Em*Sij));
Psi=phi*c;

%plotting analytical main state
subplot(2,6,1:3)
hold off
plot(x*1e9, U/e, '--k', 'LineWidth', 1)
hold on;
Amp=P(islocalmax(abs(P(:,1))),1);
for i=1:1
    plot(x*1e9, Ei(i)/e+0.1*P(:,i)/Amp,'-');
    plot(x*1e9, Ei(i)*ones(1,Np)/e,'--r');
    text(-4.5,Ei(i)/e,sprintf('$E_%i = %2.2f meV$',[i Ei(i)/e*1000]),...
        'Interpreter','latex','FontSize',14,...
        'HorizontalAlignment','left','VerticalAlignment','bottom')
end
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$E,eV$', 'Interpreter', 'latex');
title('$Numreical$ $\Psi,E$', 'Interpreter', 'latex');
E1=Ei(1);
grid on;

%plotting methods main state
subplot(2,6,4:6)
hold off
plot(x*1e9, U/e, '--k', 'LineWidth', 1)
hold on
Psi=abs(Psi');
Amp=Psi(islocalmax(abs(Psi)));
plot(x*1e9,Em/e+0.1*Psi/max(Amp))
plot(x*1e9, Em/e*ones(1,Np),'--r');
text(-4.5,Em/e,sprintf('$E_{var} = %2.2f meV$',Em/e*1000),'Interpreter','latex',...
'FontSize',14,'HorizontalAlignment','left','VerticalAlignment','bottom')
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$E,eV$', 'Interpreter', 'latex');
title('$Variated$ $\Psi,E$', 'Interpreter', 'latex');
grid on;

%plotting basis functions
subplot(2,6,11:12)
bar(c);
xlabel('$n^{th}$ $component$', 'Interpreter', 'latex');
ylabel('$C_n$', 'Interpreter', 'latex');
title('$Variated$ $\Psi,E$', 'Interpreter', 'latex');
grid on;
camroll(90)
set(gca,'YDir','reverse')

%plotting basis functions' coeffisients
subplot(2,6,9:10)
plot(ones(1,Count).*x'*1e9,2.5*phi+m)
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$\phi_n(x)$', 'Interpreter', 'latex');
title('$Basis$ $functions$ $\Psi,E$', 'Interpreter', 'latex');
grid on;
ylim([0 Count+1])

%verification of WF by comparison energies
Err = uicontrol('style','text');
txterr=sprintf('Calculation error is %2.2f%%',((Em-E1)/E1*100));
set(Err,'String',txterr,'FontSize',14,'FontWeight','bold');
Err.Units='normalized';
Err.Position=[0.15 0.20 0.2 0.1];
clear
end

function Varbutpush(~,~)
%creating a figure, where graphics and lisbox located
fig=figure('Name','Method of variations: Argument type','NumberTitle','off');
fig.Units='normalized';
fig.Position=[0 0 1 1];

%creating an interactive module, where you can choose type of force field.
%function for realizing method of variations is called here
PetrDefText=uicontrol(fig,'style','text','BackgroundColor',...
                      get(fig, 'color'),'String','Define Feild type',...
                     'FontSize',16);
PetrDefText.Units='normalized';
PetrDefText.Position=[0.4 0.9 0.2 0.05];
PetrType=uicontrol(fig, 'style','listbox',...
                   'string',{'\/-type','/\-type','cos-type','gauss-type'},...
                   'FontSize',12,'HorizontalAlignment', 'left',...
                   'Value',1,'Callback',@Variation);
PetrType.Units='normalized';
PetrType.Position=[0.4 0.8 0.2 0.1];
end

function Variation(h,~)
%defining constants
hbar=1.0546e-34;
m0=9.1e-31;
e=1.6e-19;

%forming a task
L=1e-8;                         %width of structure
Np=1000;                        %amount of steps
x=linspace(-L/2,L/2, Np);       %creating a 'x-axis'
dx=x(2)-x(1);                   %definig a primive step
koef=-hbar^2/(2*m0*12*(dx^2));  %definig a coefficient for analytical solving

%defining type of force field
type=h.Value;
switch type
    case 2
        U=e/2-abs(x/(L/2)*e/2);
    case 1
        U=abs(x/(L/2)*e/2);
    case 3
        U=e/2*(1+sin(pi*10*x/L));
    case 4
        U=e/2*(exp(-x.^2/(L/10)^2));
end

%numerical solution for hamiltonian
%defining secind devirative
E=eye(Np)*(-30);
E=E+diag(ones(1,Np-1)*16,-1);
E=E+diag(ones(1,Np-1)*16,1);
E=E+diag(ones(1,Np-2)*(-1),-2);
E=E+diag(ones(1,Np-2)*(-1),2);

%Hamiltonian
H=E*koef+diag(U);

%finding eigenvalues and eigenvectors
[P,Ei]=eig(H);
Ei=diag(Ei);

%normalization eigvectors
P=P*sqrt(1/dx);

%visualizating analytical main state
subplot(2,3,1)
hold off
plot(x*1e9, U/e, '--k', 'LineWidth', 1)
hold on;
Amp=P(islocalmax(abs(P(:,1))),1); %defining amplitude for scaling a wave-function 
Amp=max(Amp)/(Ei(1))*e;
for i=1:1
    plot(x*1e9, Ei(i)/e+P(:,i)/Amp,'-');
    plot(x*1e9, Ei(i)*ones(1,Np)/e,'--r');
    text(-4.5,Ei(i)/e,sprintf('$E_%i = %2.2f meV$',[i Ei(i)/e*1000]),...
        'Interpreter','latex','FontSize',14,...
        'HorizontalAlignment','left','VerticalAlignment','bottom')
end
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$E,eV$', 'Interpreter', 'latex');
text((sum(xlim) + diff(xlim))/2+.07*(diff(xlim)-sum(xlim))/2,0.5*sum(ylim),...
    '${\Psi\over \Psi_{1MAX}}+E$','Interpreter','latex','Rotation',-90,...
    'HorizontalAlignment','center','VerticalAlignment','baseline','FontSize',16);
title('$Numreical$ $\Psi,E$', 'Interpreter', 'latex');
E1=Ei(1);
grid on;

%solving with variation method
%for choosen Gaboure variation function we have three variables, that should be varied
%first one variable is sigma
lmbd=2*L;
mu=0;
sg=linspace(0.01*L,2*L,Np);
w=real(exp(-(x-mu).^2./(2*sg'.^2)+1i*(2*pi*(x-mu)/lmbd)));
w=w./sqrt(sum(w.*w,2)*dx);

%solving for E_med of that function
dE=zeros(Np);
for i=1:Np
    G=[0 0 w(i,:) 0 0];
    for n=3:Np-2
        Dif2=-G(n-2)+16*G(n-1)-30*G(n)+16*G(n+1)-G(n+2);
        dE(i,n)=conj(G(n))*(koef*Dif2+U(n).*G(n));
    end
end
E=real(sum(dE,2)*dx);

%choosing minimal value of energy and corresponding sigma
[~,n]=min(E);
sigma=sg(n);
Es=E(n);

%visualizating dependence of energy by sigma
subplot(2,3,4)
hold off
plot(sg*1e9,E/e,'HandleVisibility','off')
hold on
plot(sigma*1e9,Es/e,'*')
xlabel('$\sigma,nm$', 'Interpreter', 'latex');
ylabel('$E,eV$', 'Interpreter', 'latex');
title('$E(\sigma)$', 'Interpreter', 'latex');
grid on;
legend(['$\sigma=',num2str(round(sigma*1e9*100)/100),'nm$'], 'Interpreter', 'latex');

%second one variable is lambda
LMBD=linspace(0.05*L,4*L,Np)';
w=real(exp(-(x-mu).^2./(2*sigma.^2)+1i*(2*pi*(x-mu)./LMBD)));
w=w./sqrt(sum(w.*w,2)*dx);

%solving for E_med of that function
dE=zeros(Np);
for i=1:Np
    G=[0 0 w(i,:) 0 0];
    for n=3:Np-2
        Dif2=-G(n-2)+16*G(n-1)-30*G(n)+16*G(n+1)-G(n+2);
        dE(i,n)=conj(G(n))*(koef*Dif2+U(n).*G(n));
    end
end
E=real(sum(dE,2)*dx);

%choosing minimal value of energy and corresponding lambda
[~,n]=min(E);
lmbd=LMBD(n);
El=E(n);

%visualizating dependence of energy by lambda
subplot(2,3,5)
hold off
plot(LMBD*1e9,E/e,'HandleVisibility','off')
hold on
plot(lmbd*1e9,El/e,'*')
xlabel('$\lambda,nm$', 'Interpreter', 'latex');
ylabel('$E,eV$', 'Interpreter', 'latex');
title('$E(\lambda)$', 'Interpreter', 'latex');
grid on;
legend(['$\lambda=',num2str(round(lmbd*1e9)/100),'nm$'], 'Interpreter', 'latex');

%third one variable is mu
MU=linspace(-L/2,L/2,Np)';
w=real((exp(-(x-MU).^2./(2*sigma.^2)+1i*(2*pi*(x-MU)/lmbd))));
w=w./sqrt(sum(w.*w,2)*dx);

%solving for E_med of that function
dE=zeros(Np);
for i=1:Np
    G=[0 0 w(i,:) 0 0];
    for n=3:Np-2
        Dif2=-G(n-2)+16*G(n-1)-30*G(n)+16*G(n+1)-G(n+2);
        dE(i,n)=conj(G(n))*(koef*Dif2+U(n).*G(n));
    end
end
E=real(sum(dE,2)*dx);

%choosing minimal value of energy and corresponding mu
[~,n]=min(E);
mu=MU(n);
Em=E(n);
E2=Em;

%visualizating dependence of energy by mu
subplot(2,3,6)
hold off
plot(MU*1e9,E/e,'HandleVisibility','off')
hold on
plot(mu*1e9,Em/e,'*')
xlabel('$\mu,nm$', 'Interpreter', 'latex');
ylabel('$E,eV$', 'Interpreter', 'latex');
title('$E(\mu)$', 'Interpreter', 'latex');
grid on;
legend(['$\mu=',num2str(round(mu*1e9)/100),'nm$'], 'Interpreter', 'latex');

%visualizating maximal optimal wave function for given variables
Psi=real((exp(-(x-mu).^2./(2*sigma.^2)+1i*(2*pi*(x-mu)/lmbd))));
Psi=Psi./sqrt(sum(Psi.*Psi,2)*dx);
subplot(2,3,3)
hold off
plot(x*1e9, U/e, '--k', 'LineWidth', 1)
hold on
plot(x*1e9,Em/e+Psi/Amp)
plot(x*1e9, Em/e*ones(1,1000),'--r');
text(-4.5,Em/e,sprintf('$E_{var} = %2.2f meV$',Em/e*1000),'Interpreter','latex',...
'FontSize',14,'HorizontalAlignment','left','VerticalAlignment','bottom')
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$E,eV$', 'Interpreter', 'latex');
title('$Variated$ $\Psi,E$', 'Interpreter', 'latex');
grid on;

%verification of WF by comparison energies
Err = uicontrol('style','text');
txterr=sprintf('Calculation error is %2.2f%%',((E2-E1)/E1*100));
set(Err,'String',txterr,'FontSize',14,'FontWeight','bold');
Err.Units='normalized';
Err.Position=[0.4 0.7 0.2 0.1];
clear
end

function Ndegbutpush(~,~)
%creating a figure, where graphics and lisbox located
fig=figure('Name','Non-degenerated stationary theory','NumberTitle','off');
fig.Units='normalized';
fig.Position=[0 0 1 1];

%creating an interactive module, where you can choose type of force field
%or petrubation type
%function for realizing method of variations is called here
PetrDefText=uicontrol(fig,'style','text','BackgroundColor',...
                      get(fig, 'color'),'String','Define feild type',...
                     'FontSize',16);
PetrDefText.Units='normalized';
PetrDefText.Position=[0.15 0.35 0.3 0.1];
PetrType=uicontrol(fig, 'style','listbox',...
                   'string',{'\/-type','/\-type','cos-type','gauss-type','rect-type',...
                   'Upetr-sin','Upetr-step'},...
                   'FontSize',12,'HorizontalAlignment', 'left',...
                   'Value',1,'Callback',@Ndegenerated);
PetrType.Units='normalized';
PetrType.Position=[0.15 0.3 0.3 0.1];
end

function Ndegenerated(h,~)

%defining constants
hbar=1.0546e-34;
m0=9.1e-31;
e=1.6e-19;

%forming a task
L=1e-8;                         %width of structure
Np=1000;                        %amount of steps
x=linspace(-L/2,L/2, Np);       %creating a 'x-axis'
dx=x(2)-x(1);                   %definig a primive step
koef=-hbar^2/(2*m0*12*(dx^2));  %definig a coefficient for analytical solving

%defining potential feild
U=zeros(1,Np);
Upetr=[zeros(1,Np/2), ones(1,Np/2)*0.01*e];
%these fields are considered to be default
type=h.Value;
if type>0 || type<8
    switch type
        case 2
            U=e/2-abs(x/(L/2)*e/2);
        case 1
            U=abs(x/(L/2)*e/2);
        case 3
            U=e/2*(0.5+0.5*sin(pi*10*x/L));
        case 4
            U=e/2*(exp(-x.^2/(L/10)^2));
        case 5
            U=zeros(1,Np);
        case 6
            Upetr=0.01*e*(1+sin(pi*10*x/L));
        case 7
            Upetr=[zeros(1,Np/2), ones(1,Np/2)*0.01*e];
    end
end

%numerical solution for hamiltonian
%defining secind devirative
E=eye(Np)*(-30);
E=E+diag(ones(1,Np-1)*16,-1);
E=E+diag(ones(1,Np-1)*16,1);
E=E+diag(ones(1,Np-2)*(-1),-2);
E=E+diag(ones(1,Np-2)*(-1),2);

%Hamiltonian
H=E*koef+diag(U);
Hpetr=E*koef+diag(U+Upetr);

%finding eigenvalues and eigenvectors
[P,En]=eig(H);
[Ppetr,Enpetr]=eig(Hpetr);
En=diag(En);
Enpetr=diag(Enpetr);

%Finding an amendment given by petrubation
n=P(:,1);
dE1=n'*diag(Upetr)*n/e; %first attempt to energy
dn=0; dE2=0;
for i=2:10              %first attempt to function and second for energy
    k=P(:,i);
    dn=dn+(k'*diag(Upetr)*n)./(En(1)-En(i))*k;
    dE2=dE2+(k'*diag(Upetr)*n)./(En(1)-En(i))*(n'*diag(Upetr)*k);
end 
dE2=dE2/e;%J->eV

%visualization
%non-petrubated state
subplot(2,2,1)
hold off
plot([-L/2-dx -L/2 x L/2 L/2+dx]*1e9, [0.25 0.25 U/e 0.25 0.25], '--k', 'LineWidth', 1)
hold on;
[~,zx]=max(abs(P(:,1)));
amp=P(zx,1);%defining amplitude for scaling a wave-function
for i=1:1
    plot(x*1e9, En(i)/e+0.05*P(:,i)/amp);
    plot(x*1e9, En(i)*ones(1,Np)/e,'--r');
end
E1=En(1)/e;%J->eV
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$E,eV$', 'Interpreter', 'latex');
title('$\Psi_0,E$', 'Interpreter', 'latex');
text((sum(xlim) + diff(xlim))/2-0.2*(diff(xlim)-sum(xlim))/2,En(1)/e,...
    sprintf('$E_0 = %2.2f meV$',E1*1000),'Interpreter','latex',...
    "HorizontalAlignment",'right','VerticalAlignment','bottom')
grid on;
ylim([0 2*(E1+0.05)])
xlim([-L/2*1.1 L/2*1.1]*1e9)

subplot(2,2,2)
hold off
plot([-L/2-dx -L/2 x L/2 L/2+dx]*1e9, [0.25 0.25 Upetr/e 0.25 0.25], '--k', 'HandleVisibility','off')
hold on;
%pseudoanalytical solution
[~,zx]=max(abs(Ppetr(:,1)));%defining amplitude for scaling a wave-function
amp=Ppetr(zx,1);
plot(x*1e9, Enpetr(1)/e+0.05*Ppetr(:,1)/amp(1),'go','LineWidth', 1.5, 'HandleVisibility','on');
P=n+dn;
%petrubation theory solution
amp=(max(abs(P)));
plot(x*1e9, (En(1)/e+dE1)+0.05*(abs(P))/amp,'r');
legend('analytical','petrubated','Interpreter', 'latex');
plot(x*1e9, Enpetr(1)*ones(1,Np)/e,'--r', 'HandleVisibility','off');
E2=Enpetr(1)/e;
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$E,eV$', 'Interpreter', 'latex');
title('$\Psi_{petr},E$', 'Interpreter', 'latex');
grid on;
ylim([0 2*(E1+0.05)])
text((sum(xlim) + diff(xlim))/2-0.5*(diff(xlim)-sum(xlim))/2,0.7*sum(ylim),...
     sprintf("$E = %2.2f meV$ \n $e_1' = %2.2f meV$\n $e_2' = %2.2f meV$",[E2,dE1,dE2]*1000),'Interpreter','latex')
xlim([-L/2*1.1 L/2*1.1]*1e9)

%verification of WF by comparison energies
Err = uicontrol('style','text');
txterr=sprintf("Estimated Energy in pуtrubation: E= Eo +e1+e2=%2.2fmeV\nCalculation error is %2.2f%%",...
    [(E1+dE1+dE2)*1e3,abs(((-E2+E1+dE1+dE2)/E2)*100)]);
set(Err,'String',txterr,'FontSize',14,'FontWeight','bold');
Err.Units='normalized';
Err.Position=[0.15 0.2 0.3 0.1];
end

function Degbutpush(~,~)
%creating a figure, where graphics and aliders are located
fig=figure('Name','Degenerated stationary theory: 2D-hole','NumberTitle','off');
fig.Units='normalized';
fig.Position=[0 0 1 1];

%creating an interactive module, where you can set lambda - coupling coefficient
pnl = uipanel(fig);
pnl.Units='normalized';
pnl.Position=[0.405 0.475 0.22 0.08];
PetrDefText=uicontrol(fig,'style','text','BackgroundColor',...
                      get(fig, 'color'),'String','Define coefficient lambda',...
                     'FontSize',16);
PetrDefText.Units='normalized';
PetrDefText.Position=[0.41 0.5 0.21 0.05];

%at first lmbd set as zero
lmbd=0;
Degenerated(0,0);
%creating a slider to vary lambda value
PetrCoef=uicontrol(fig, 'style','slider',...
                   'min',0,'max',0.5,'SliderStep',[0.05 0.05],...
                   'Value',lmbd,'callback',@Degenerated);
PetrCoef.Units='normalized';
PetrCoef.Position=[0.41 0.5 0.21 0.02];
end

function Degenerated(h,~)
%defining constants
hbar=1.0546e-34;
m0=9.1e-31;
e=1.6e-19;

%forming a task
L=1e-8;                         %width of structure
Np=100;                         %amount of steps
x=linspace(0,L, Np);            %creating a 'x-axis'
dx=x(2)-x(1);    dy=dx;         %definig a primive step
y=x';                           %creating a 'y-axis'
[X,Y]=meshgrid(x,y);            %creating a 2d-sapce

%setting a lambda(h may have float value or uicontrol value)
if h==0
    lmbd=0;
else
    lmbd=h.Value;
end

%defining potential feild and perubation
U=zeros(Np);
Upetr=lmbd*e*heaviside(-x+L/4).*heaviside(-y+L/4);

%defining wave functions for quantumm well
px=@(n,x)sqrt(2/L).*sin(pi*n*x/L);
py=@(m,y)sqrt(2/L).*sin(pi*m*y/L);
p=@(n,m,x,y)px(n,x).*py(m,y);
E=@(n,m)pi^2*hbar^2/(2*m0*L^2)*(n^2+m^2);

%set n=1 and m=2 as it is degenerated w/ n=2 and m=1
Psi1=p(1,2,X,Y);
E1=E(1,2);
Psi2=p(2,1,X,Y);
E2=E(2,1);

%Finding an amendment to Energy
%we know that states 2:1 and 1:2 are degenerated and have same energy 
%that becomes unequal after petrubation

%Vab=<Psia|Upetr|Psib>; effect of eptrubation
V11=sum(sum(Psi1'.*Upetr.*Psi1,2)*dx,1)*dy;
V12=sum(sum(Psi1'.*Upetr.*Psi2,2)*dx,1)*dy;
V21=sum(sum(Psi2'.*Upetr.*Psi1,2)*dx,1)*dy;
V22=sum(sum(Psi2'.*Upetr.*Psi2,2)*dx,1)*dy;
%as we can see all of them have some non-zero value, but we want some to be

%solving a task to find new 'good' eigfunctions
M=[V11, V12; V21,V22];
[ab,de]=eig(M);
de=diag(de);%energeis for 'good' states

%new 'good' eigfunctions
%Xi=ab*[Psi1;Psi2];
if(de(1)>de(2))
    Xi1=ab(1,1)*Psi1+ab(1,2)*Psi2;
    Xi2=ab(2,1)*Psi1+ab(2,2)*Psi2;
else
    Xi2=ab(1,1)*Psi1+ab(1,2)*Psi2;
    Xi1=ab(2,1)*Psi1+ab(2,2)*Psi2;
    temp=de(1);
    de(1)=de(2);
    de(2)=temp;
end

%visualization 
%(2;1)initial state
subplot(2,3,1)
Amp=max(max(Psi1));
hold off
surf(X*1e9, Y*1e9, U/e,'FaceAlpha',0.5)
hold on;
surf(X*1e9, Y*1e9,E1/e+0.25*Psi1/Amp);
surf(X*1e9, Y*1e9,E1*ones(Np)/e);
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$y,nm$', 'Interpreter', 'latex');
zlabel('$E+{\Psi\over\Psi_{max}}$', 'Interpreter', 'latex');
title(sprintf('$Psi(2,1),E=%2.2f$ $meV$',E1/e*1000), 'Interpreter', 'latex');
grid on;
shading interp;

%force feild
subplot(2,3,2)
hold off
surf(X*1e9, Y*1e9, (U+Upetr)/e)
hold on;
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$y,nm$', 'Interpreter', 'latex');
zlabel('$U, meV$', 'Interpreter', 'latex');
title('$Pertubated$ $potential$ $field$', 'Interpreter', 'latex');
grid on;
shading interp;
zlim([-1 1])

%(1;2)initial state
subplot(2,3,3)
hold off
surf(X*1e9, Y*1e9, U/e,'FaceAlpha',0.5)
hold on;
surf(X*1e9, Y*1e9,E2/e+0.25*Psi2/Amp);
surf(X*1e9, Y*1e9,E2*ones(Np)/e);
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$y,nm$', 'Interpreter', 'latex');
zlabel('$E+{\Psi\over\Psi_{max}}$', 'Interpreter', 'latex');
title(sprintf('$Psi(1,2),E=%2.2f$ $meV$',E2/e*1000), 'Interpreter', 'latex');
grid on;
shading interp;

%1 petr state, Xi1
subplot(2,3,4)
set(0,'defaulttextInterpreter','latex')
hold off
surf(X*1e9, Y*1e9, (U)/e,'FaceAlpha',0.5)
hold on;
surf(X*1e9, Y*1e9,(E1+de(1))/e+0.25*Xi1/Amp);
% surf(X*1e9, Y*1e9,E1*ones(Np)/e);
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$y,nm$', 'Interpreter', 'latex');
zlabel('$E,{\Psi\over\Psi_{max}}$', 'Interpreter', 'latex');
title(sprintf('$X_1,E=%2.2f$ $meV$',(E1+de(1))/e*1000), 'Interpreter', 'latex');
grid on;
shading interp;

%dependences E1(lmbd) and E2(lmbd)
subplot(2,3,5)
LMBD=linspace(0,0.5, Np);
plot(LMBD,ones(1,Np)*E1/e*1000,'--k','HandleVisibility','off')
hold on
plot(lmbd,(E1+de(1))/e*1000,'ob')
plot(lmbd,(E2+de(2))/e*1000,'or')
xlabel('$\lambda$', 'Interpreter', 'latex');
ylabel('$E,meV$', 'Interpreter', 'latex');
title('$E_1(\lambda),E_2(\lambda)$', 'Interpreter', 'latex');
text(0.5,5+E1/e*1000, sprintf('$E_1 = E_2 = %2.2f meV$',E1/e*1e3),...
    'Interpreter','latex','HorizontalAlignment','right','VerticalAlignment','baseline')
legend("E1'","E2'")
grid on;
ylim([0 0.05]*1000)
xlim([0 0.5])

%2 petr state,%1 petr state, Xi1
subplot(2,3,6)
hold off
surf(X*1e9, Y*1e9, (U)/e,'FaceAlpha',0.5)
hold on;
surf(X*1e9, Y*1e9,(E2+de(2))/e+0.25*Xi2/Amp);
% surf(X*1e9, Y*1e9,E2*ones(Np)/e);
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$y,nm$', 'Interpreter', 'latex');
zlabel('$E,{\Psi\over\Psi_{max}}$', 'Interpreter', 'latex');
title(sprintf('$X_2,E=%2.2f$ $meV$',(E2+de(2))/e*1000), 'Interpreter', 'latex');
grid on;
shading interp;

%interfacing lambda-value
Lambda=uicontrol('style','text','String',['lambda = ', num2str(lmbd)],...
    'FontSize',14,'FontAngle','italic','HorizontalAlignment','center');
Lambda.Units='normalized';
Lambda.Position=[0.41 0.48 0.21 0.02];
end

function Nonstbutpush(~,~)
%creating a figure, where graphics and lisbox located
fig=figure('Name','Nonstationary theory: Quantum well','NumberTitle','off');
fig.Units='normalized';
fig.Position=[0 0 1 1];

%definig initial time interval as 100 seconds
T=0;
GoldenFermiRule(T,0);

%creating a slider to vary time value
NstatTime=uicontrol(fig, 'style','slider',...
                   'min',-34,'max',34,'SliderStep',[0.005 0.005],...
                   'Value',T,'callback',@GoldenFermiRule);
NstatTime.Units='normalized';
NstatTime.Position=[0.41 0.5 0.21 0.02];
 end

function GoldenFermiRule(h,~)
%defining constants
hbar=1.0546e-34;
m0=9.1e-31;
e=1.6e-19;

%forming a task
L=1e-8;
Np=1001;
x=linspace(0,L, Np)';

%setting Time-value(h may have float value or uicontrol value)
if h==0
    T=10^(0);
else
    T=10^(h.Value);
end

%defining potential feild and perubation
%U=zeros(Np);                       %inf quantum well
Upetr=diag(ones(1,Np)*0.5*e);       %consdered as multiplied by exp(-iwt)
p=@(n)sqrt(2/L).*sin(pi*n*x/L);     %n-th state
E=@(n)pi^2*hbar^2/(2*m0*L^2)*(n^2); %n-th energy

%defining states
a=1;  b=2;
p1=p(a); E1=E(a);
p2=p(b); E2=E(b);
w=(E2-E1)/hbar;                     %allowed frequency for a->b
W=linspace(w*0.9, 1.1*w, Np);           %araay of frequncies 

%defining golden fermi rule function
D=@(E,t)4*sin(E.*t/(hbar*2)).^2./E.^2;
d=@(E,t)hbar./(2*pi*t).*D(E,t);
P=@(w,t)abs(p2'*Upetr*p1)^2.*(d(E2-E1-w*hbar,t));
P1=@(w,t)abs(p2'*Upetr*p1)^2*hbar./(2*pi*t);
% P=@(w,t)abs(p2'*Upetr*p1)^2.*2*pi*t/hbar;
G=@(t)P(W,t)/t;

%visualization 
%a and b states, considered transmition from a to b
subplot(2,2,1)
hold off
grid on
hold on
for i=1:(max(a,b)+2)
    plot([0 L]*1e9, E(i)*[1 1]/e,'--')
end
plot([0 L]*1e9, [1 1]*E1/e,'--r')
plot([0 L]*1e9, [1 1]*E2/e,'--b')
plot(x*1e9, E1/e+p1/max(p1)/100,'r')
plot(x*1e9, E2/e+p2/max(p2)/100,'b')
plot([0.3 0.3], [E1 E2]/e,'b','Color', [0 0 0],'LineWidth', 3)
h=annotation('arrow');
set(h,'parent', gca,'position', [0.3 E1/e 0 (E2-E1)/e],'HeadLength', 10,...
    'HeadWidth', 10, 'HeadStyle', 'hypocycloid','Color', [0 0 0], 'LineWidth', 0.5);
xlabel('$x,nm$', 'Interpreter', 'latex');
ylabel('$E,eV + \Psi$', 'Interpreter', 'latex');
title('$States$ $of$ $quantumm$ $well$', 'Interpreter', 'latex');

%dependence of transition by time
t=linspace(0,T);
subplot(2,2,2)
hold off
plot(t,P1(w,t))
hold on
grid on
xlabel('$T,sec$', 'Interpreter', 'latex');
ylabel('$P=|<b|V|a>|^2D_t(E_b-E_a-\hbar\omega)$', 'Interpreter', 'latex');
title('$Probability$ $by$ $time$', 'Interpreter', 'latex');

%dependence of probability by frequency
subplot(2,2,3)
hold off
plot(W,P(W,T))
hold on
grid on
xline(w,'--')
xlabel('$w,{rad\over \sec}$', 'Interpreter', 'latex');
ylabel('$P=|\Psi|^2$', 'Interpreter', 'latex');
title('$Probability$ $by$ $frequency$', 'Interpreter', 'latex');

%dependence of rate of transition by frequency
subplot(2,2,4)
hold off
plot(W,G(T))
hold on
grid on
xline(w,'--')
xlabel('$w,{rad\over \sec}$', 'Interpreter', 'latex');
ylabel('$\Gamma={|\Psi|^2\over t}$', 'Interpreter', 'latex');
title('$Transmission$ $rate$', 'Interpreter', 'latex');

%interfacing time-value
Timeis=uicontrol('style','text','String',['t = ', num2str(T),' seconds'],...
    'FontSize',14,'FontAngle','italic','HorizontalAlignment','center');
Timeis.Units='normalized';
Timeis.Position=[0.41 0.48 0.21 0.02];
end