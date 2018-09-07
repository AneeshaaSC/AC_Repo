class parent
{
	float a,b;
	float arithmetic()
	{
		return a/b;
	}
}
class child extends parent
{
	float arithmetic()
	{
		return a%b;
	}
}
public class ans3 {

	public static void main(String args[])
	{
		parent p = new parent();
		p.a=7;
		p.b=9;
		System.out.println("Division result:"+p.arithmetic());
		child c = new child();
		c.a=7;
		c.b=9;
		System.out.println("Modulus result:"+c.arithmetic());
		
	}
}
