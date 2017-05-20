% 将wav文件转化为音符
% 王旭康，2016.5.27
% 适用Matlab2014以上

% 注意，测试所有的歌曲都已经调整到了每分钟80.7拍，以适应FFT的需要
% 将乐曲分帧，分别进行傅里叶变换
    % 以取样频率 f0=44100Hz 计算
    % 一帧包含16384个数据时，即持续0.372s；频率精度为2.69Hz
    % 此文件进行的是单音符识别，取得范围内最强频率，再向下查找2~4倍找到基频

% 请在这里设置文件名
file = 'piano';

wav_file = strcat(file,'.wav');
output_file_name = strcat(file,'.txt');
[SIGNAL,FREQ]=audioread(wav_file); 

LEN=16384;  %傅里叶变换取样长度，即一帧包含的数据量
MAXF=700;   %最高识别频率
MINF=180;   %最低识别频率
MINA=10;    %最低响度

FREQ_LIST=linspace(0,FREQ/2,LEN/2); %获得0到FREQ/2, 共LEN/2个数字的等差数列

n=int32(length(SIGNAL)/LEN); %总帧数

notes_list = zeros(1, n);       
freqs_list = zeros(1, n);
maxA_list = zeros(1, n);    %记录最大响度

MAX_INDEX=int32(LEN/FREQ*MAXF); %最高识别频率对应的个数
MIN_INDEX=int32(LEN/FREQ*MINF); %最低识别频率对应的个数

key=['G3 ';'G3#';'A3 ';'A3#';'B3 '];
key=[key;'C4 ';'C4#';'D4 ';'D4#';'E4 ';'F4 ';'F4#';'G4 ';'G4#';'A4 ';'A4#';'B4 '];
key=[key;'C5 ';'C5#';'D5 ';'D5#';'E5 ';'F5 ';'F5#';'G5 ';'G5#';'A5 ';'A5#';'B5 '];
key=[key;'C6 ';'C6#';'D6 ';'D6#';'E6 ';'F6 ';'F6#';'G6 ';'G6#';'A6 ';'A6#';'B6 '];

%分段进行傅里叶分析
for i = 1:n, 
    lX = fft( SIGNAL( (i-1)*LEN+1 : i*LEN ) );  %截取长度为l的一段数据的快速傅里叶变换    
    lX_cut = lX(1:MAX_INDEX);                   %高频截止
    
    lA = sqrt(lX_cut.*conj(lX_cut));      %取模
    if lA > 0
        lA = 20*log10(lA);        %dB
    end
    
    max_index = MIN_INDEX;
    maxA_list(i) = lA(MIN_INDEX);
    
    for t=MIN_INDEX:MAX_INDEX
        if (lA(t)>maxA_list(i))
            maxA_list(i) = lA(t);
            max_index = t;
        end
    end
    
    best_index = max_index;
    for mult=2:4,                  %再向下查找2~4倍找到基频
        test_index = int32(max_index/mult);
        if (test_index>MIN_INDEX)
            if (lA(test_index) > maxA_list(i)*0.9) 
                best_index = int32(max_index/mult);
            end
        else
            break;            
        end
    end
    freqs_list(i) = FREQ_LIST( best_index );    %最优响度对应的频率
    notes_list(i) = log(freqs_list(i)/220) / log(2) * 12 + 3; %计算音高
    
    if maxA_list(i)<MINA || notes_list(i)<-12
        notes_list(i)=NaN;
    end
end

% 去除响度太低的点
for i = 1:n,
    if maxA_list(i)<MINA || notes_list(i)<-12
        notes_list(i)=NaN;
    end
end

% 输出到文件
output_file = fopen(output_file_name,'w');
for i=1:n
    if notes_list(i)>0
        fprintf(output_file,'%d\r\n',round(notes_list(i)));
    else
        fprintf(output_file,'0\r\n');
    end
end
fclose(output_file);

K=[0,0,0,0,0,0,0,0,0,0,0,0];  %辅助定调1

%输出原乐谱
fprintf('钢琴音：\n');
for i=1:n,
    %try
        if notes_list(i)==notes_list(i) % not NaN
            p = round(notes_list(i));
            if p>0
                for j = 1:3
                        fprintf('%c',key(p,j));
                end
            else
                fprintf('%d ',p);
            end
            fprintf(' ');
            K(mod(p-1,12)+1)=K(mod(p-1,12)+1)+1;
        else
            fprintf('??? ');
        end
    if mod(i,16)==0 
        fprintf('\n'); 
    end
end

h7=[0,2,4,5,7,9,11];
w7=[3,2,3,1,3,3,1];
maxSc=0;  % 用于统计定调
bestMc=6;
for mc=6:17,
    sc=0;
    for i=1:length(h7)
        sc=sc+K(mod(mc+h7(i)-1,12)+1)*w7(i);
    end
    if maxSc<sc
        maxSc=sc;
        bestMc=mc;
    end
end

%输出修订乐谱
fprintf('\n\n简谱：\n1=');
for j=1:3
    fprintf('%c',key(bestMc,j));
end
fprintf('\n');
for i=1:n,
    try
        if notes_list(i)==notes_list(i) 
            p=notes_list(i);            
            p=p-bestMc;
            
            while p<-0.5
                fprintf('_');
                p=p+12;
            end
            while p>=11.5
                fprintf('^');
                p=p-12;
            end            
            
            if p<1
                fprintf('1');                
            elseif p<3
                fprintf('2');
            elseif p<4.5
                fprintf('3');
            elseif p<6
                fprintf('4');
            elseif p<8
                fprintf('5');
            elseif p<10
                fprintf('6');
            else
                fprintf('7');
            end
            
            fprintf('\t');
            
            K(mod(p-1,12)+1)=K(mod(p-1,12)+1)+1;
        else
            fprintf('??\t');
        end
    catch
    end
    if mod(i,16)==0 
        fprintf('\n'); 
    end
end

fprintf('\n音频分析已完成。\n\n');

%subplot(2,2,1)
%plot(notes_list)
%subplot(2,2,3)
%plot(SIGNAL)
%subplot(2,2,2)
%hist(notes_list)