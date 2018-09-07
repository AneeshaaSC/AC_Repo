class player
{
	String name;
	String age;
	player(String name, String age)
	{
		this.name=name;
		this.age=age;
	}
}
class hockeyplayer extends player
{
	String sport;
	String position;
	hockeyplayer(String name, String age,String sport,String position)
	{
		super(name,age);
		this.sport=sport;
		this.position=position;
	}
}
class cricketplayer extends player
{
	String sport;
	String position;
	cricketplayer(String name, String age, String sport,String position)
	{
		super(name,age);
		this.sport=sport;
		this.position=position;
	}
}
class footballplayer extends player
{
	String sport;
	String position;
	footballplayer(String name, String age,String sport,String position)
	{
		super(name,age);
		this.sport=sport;
		this.position=position;
	}
}
public class ans2 {
	public static void main(String args[])
	{
		hockeyplayer h=new hockeyplayer("Dhyan Chand","35","Hockey","Forward");
		cricketplayer c=new cricketplayer("Virat Kohli","29","Cricket","Batsmen");
		footballplayer f=new footballplayer("Iker Casillas","36","Football","Goalie");
		System.out.println("Hockey player's details:");
		System.out.println("Name: "+h.name+"\n"+"Age: "+h.age+"\n"+"Sport: "+h.sport+"\n"+"Position: "+h.position+"\n\n");
		System.out.println("Cricket player's details:");
		System.out.println("Name: "+c.name+"\n"+"Age: "+c.age+"\n"+"Sport: "+c.sport+"\n"+"Position: "+c.position+"\n\n");
		System.out.println("Football player's details:");
		System.out.println("Name: "+f.name+"\n"+"Age: "+f.age+"\n"+"Sport: "+f.sport+"\n"+"Position: "+f.position+"\n\n");
		
	}

}
