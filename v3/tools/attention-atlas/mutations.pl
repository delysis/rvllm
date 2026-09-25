#!/usr/bin/env perl
# Host-only mutant preparation; no model/device or repository mutation.
use strict;
use warnings;
use File::Path qw(make_path);
use JSON::PP;
my ($source, $cpp, $out) = @ARGV;
die "usage: mutations.pl HEADER CPP NEW_ABSOLUTE_DIR\n" unless @ARGV == 3 && $out =~ m{^/} && !-e $out;
sub slurp { my ($p)=@_; open my $f,'<:raw',$p or die "$p: $!"; local $/; return <$f>; }
my $header=slurp($source);my $prefix=slurp($cpp);
$prefix =~ s/int main\(int argc,char\*\* argv\).*\z//s or die "host main boundary missing";
my @mutants=(
 ['dpanel-overwrite','scores[row*BK+j] += dot;','scores[row*BK+j] = dot;', 'run_case<512,16,8,64,128,true>(1,17,16,0);'],
 ['average-splits','precise::exp(partial[base+uint(lane)*(p.dim+2u)]-m)','1.0f/l','run_case<512,8,8,128,64,true>(1,17,16,0,4);'],
 ['future-key-leak','return key <= pos && key >= atlas_start(pos, window);','return key <= pos + 1u && key >= atlas_start(pos, window);','run_case<256,8,16,128,64,false>(3,17,14,0);'],
 ['window-off-by-one','pos + 1u - window : 0u','pos + 2u - window : 0u','run_case<256,16,8,64,128,true>(1,1031,1024,5);'],
 ['raw-v-alias','atlas_round(atlas_widen(k[index]) * factor[physical_token]);','k[index];','run_case<512,16,8,64,128,true>(1,9,8,0,4,true);'],
);
make_path($out);
sub put { my ($p,$s)=@_;open my $f,'>:raw',$p or die "$p: $!";print {$f} $s;close $f or die $!; }
for my $m (@mutants) {
 my ($name,$old,$new,$body)=@$m;
 my $h=$header; my $n=($h =~ s/\Q$old\E/$new/g);die "expected one site for $name, got $n" unless $n==1;
 my $dir="$out/$name";make_path($dir);
 put("$dir/atlas_under_test.hpp",$h);
 put("$dir/test.cpp",$prefix."int main(){try{$body return 0;}catch(const std::exception& e){std::cerr<<\"FAIL: \"<<e.what()<<std::endl;return 1;}}\n");
 put("$dir/mutation.json",JSON::PP->new->canonical->pretty->encode({name=>$name,old=>$old,new=>$new,case=>$body,scope=>'host-emulation-only'}));
}
