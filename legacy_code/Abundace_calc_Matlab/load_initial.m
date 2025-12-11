function v0 = load_initial(previous, sizes)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = sizes(1);
y = sizes(2);

v0 = double(zeros(x,y,5));

if exist(previous) == 2
	eggs = ncArray(previous, 'eggs');
	eggs_dia = ncArray(previous, 'ed');
	juv = ncArray(previous, 'juv');
	imm = ncArray(previous, 'imm');
	mat = ncArray(previous, 'adults');
	v0(:,:,1) = eggs(:,:,end);
	v0(:,:,2) = eggs_dia(:,:,end);
	v0(:,:,3) = juv(:,:,end);
	v0(:,:,4) = imm(:,:,end);
	v0(:,:,5) = mat(:,:,end);
else
	v0(:,:,2) = 625.0*100.0*ones(x,y);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
