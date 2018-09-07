package assignment4;

class Volume_of_box
{
	float w;
	float l;
	float h;

	void vol_calc()
	{
		w=12;
		l=12;
		h=12;
		float vol=w*l*h;
		System.out.println("Volume of the box = "+vol+" cm3");
	}

}

public class ans3 {


	public static void main(String args[])
	{
		Volume_of_box b1=new Volume_of_box();
		b1.vol_calc();

			
	}

}
