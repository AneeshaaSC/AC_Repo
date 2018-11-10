% Read images
lowim=imread("marilyn.bmp"); %low frequency image
hiim=imread("einstein.bmp"); %high frequency image


disp("size of low freq image:");
disp(size(lowim));

disp("size of high freq image:");
disp(size(hiim));

%pad images, pad size=30
pad=30; 
% each image will now have an additional 60 rows (30 appended to the top and 30 appended to the bottom) and
% 60 additional columns (30 appended to the left and 30 appended to the % right)

% pad images to maintain detail/information on image borders
padded_loim=padarray(lowim,[pad,pad],'both'); %pad low frequency image 
disp('size of padded low frequency image:');   
disp(size(padded_loim));

padded_hiim=padarray(hiim,[pad,pad],'both');%pad high frequency image
disp('size of padded high frequency image:');
disp(size(padded_hiim));


%filter for low-frequency/marilyn image
sigma_lo = 3; % sigma/variance 
fil_lo=zeros(floor(8*sigma_lo+1),floor(8*sigma_lo+1)); % compute size of filter by convention
fil_lo= make_size_odd(fil_lo);% ensure filter size is odd
 

%filter for high-frequency/einsteins image
sigma_hi = 7;% sigma/variance 
fil_hi=zeros(floor(8*sigma_hi+1),floor(8*sigma_hi+1));  % compute size of filter by convention
fil_hi=make_size_odd(fil_hi); % ensure filter size is odd
 

% create gaussian filters 
template_lo = make_gauss_filter(fil_lo,sigma_lo); %for low frequency image (marilyn)
template_hi = make_gauss_filter(fil_hi,sigma_hi); %for high frequency image(einstein)
 

% convert class of input image matrix to double type, so that matlab
% supports arithmetic operations between original image matrix and
% convolved matrix
padded_hiim=im2double(padded_hiim);
padded_loim=im2double(padded_loim);
 

% convolve template and image
temp_hifreq_im=conv_gauss_filter(padded_hiim,template_hi);
lowfreq_im=conv_gauss_filter(padded_loim,template_lo);

disp("size of high frequency convolved image");
disp(size(temp_hifreq_im));
disp("size of low frequency convolved image");
disp(size(lowfreq_im));


% % obtain image with high-frequency components by subtracting low-pass
% % filtered image from original image
hifreq_im=padded_hiim-temp_hifreq_im;
 
 
hyb=hifreq_im+lowfreq_im;
 
% % display intermediate results along with final hybrid image
subplot(2,3,1), imshow(padded_hiim);
subplot(2,3,2), imshow(temp_hifreq_im);
subplot(2,3,2), imshow(hifreq_im);
subplot(2,3,4), imshow(padded_loim);
subplot(2,3,5), imshow(lowfreq_im);
subplot(2,3,6), imshow(hyb);
  

%used functions

%make 
%make filter size odd, if size of any dimension is even
function filter= make_size_odd(fil)
    if rem(size(fil,1),2)==0
        fil=zeros(size(fil,1)+1,size(fil,2));
    end
    
    if rem(size(fil,2),2)==0
        fil=zeros(size(fil,1),size(fil,2)+1);
    end
    filter=fil;
end

%convolution function
function convolved_image = conv_gauss_filter(input_image,template)
    %padd zeros to template before convolving, in case tempalte is
    %asymmetric. Then flip along both dimensions to achieve convolution
    %no need to flip kernel if kernel is symmetric
    if size(template,1)>size(template,2)
        template=padarray(template,[0,(size(template,1)-size(template,2))],'post');
        template=fliplr(template);% flip along vertical axis
        template=flipud(template);% flip along horizontal axis
    else
        template=padarray(template,[(size(template,2)-size(template,1)),0],'post');  
        template=fliplr(template);
        template=flipud(template);
    end
    
    % store size of images
    [irows, icols, ich] = size(input_image); %image rows, image columns, image channels(3 in number of RGB/color image)

    %initialize temporary black image
    temp(1:irows, 1:icols,1:ich)=0;
    %store filter sizes
    [trows,tcols]=size(template);

    trhalf=floor(trows/2); %rounded value of half of the rows in the template
    tchalf=floor(tcols/2); %rounded value of half of the cols in the template
    
    % perform convolution
    for c=1:ich
        for x=trhalf+1:icols-trhalf %address all columns except border
            for y=tchalf+1:irows-tchalf %address all columns except border
                sum=0;
                for iwin=1:trows
                    for jwin=1:tcols
                        sum=sum+input_image(y+jwin-tchalf-1, x+iwin-trhalf-1,ich) *template(jwin,iwin);
                    end
                end
                temp(y,x,c)=sum;
            end
        end
    end
    convolved_image = temp; 
end

%template creation function
function filter = make_gauss_filter(fil,sigma)
    filter=zeros(size(fil));
    rows=size(fil,1);
    cols=size(fil,2);
    %determine the centre of the gaussian filter, so that weights vary relative to
    %centre
    cx=floor(size(fil,1)/2)+1;
    cy=floor(size(fil,2)/2)+1;
    %sum of filter weights
    sum=0;
    for j= 1:cols
        for i=1:rows
            filter(i,j)=exp(-(((j-cy)^2+(i-cx)^2)/(2*sigma*sigma)));
            sum=sum+filter(i,j);
        end
    end

%normalize the filter
    filter=filter/sum;
end