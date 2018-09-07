package assignment4;

class Rectangle
{
	float w;
	float l;
	float r_area;
	String col;
	void set_dimensions(float len,float wid)
	{
		l=len;
		w=wid;
	}
	void set_area()
	{
		r_area=w*l;
	}
	void set_color(String color)
	{
		col=color;
	}
	void printing()
	{
		System.out.println("length="+l+"\nwidth="+w+"\narea="+r_area+"\ncolor="+col);
	}
}

public class ans2 {
	public static void main(String args[])
	{
		Rectangle r1=new Rectangle();
		r1.set_dimensions(15, 5);
		r1.set_area();
		r1.set_color("blue");
		r1.printing();
		Rectangle r2=new Rectangle();
		r2.l=30;
		r2.col="yellow";
		r2.w=12;
		r2.printing();
			
	}

}
