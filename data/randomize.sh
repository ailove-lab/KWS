#!/usr/bin/fish
for d in (ls ideal)
    mkdir -p train/$d
    for f in ideal/$d/*
        for i in (seq 4)
            set r (random 0 400)
            set pitch (math "$r-200"
            set r (random 0 100)
            set tempo (math -s2 "1.2
            set r (random 5 80)
            set vol (math -s2 "$r/10
            set out train/$d/(basena
            sox $f $out pitch $pitch
        end
    end
end
