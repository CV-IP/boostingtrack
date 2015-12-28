function angle = interangle(A,B)
if norm(A-B) < 1e-10
    angle = 0;
else
    angle = acos(dot(A,B)/(norm(A)*norm(B)))*180/pi;
end
end
